import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models.modules import PytorchClassifierAsSklearnClassifier


class SimpleCFNode:
    def __init__(
        self, max_instances, is_leaf=True, min_samples_for_update=10, node_index=0, args=None
    ):
        self.is_leaf = is_leaf
        self.children = []
        self.data_points = []
        self.labels = []
        self.max_instances = max_instances
        self.centroid = None
        self.classifier = None
        self.classes = []
        self.min_samples_for_update = min_samples_for_update
        self.new_data_points = 0
        self.node_index = node_index
        self.new = True
        self.args = args

    def update_centroid(self, data_point):
        data_point = np.array(data_point)
        if self.centroid is None:
            self.centroid = data_point
        else:
            n = len(self.data_points)
            self.centroid = (self.centroid * (n - 1) + data_point) / n

    def train_classifier_essentials(self):
        X = np.array(self.data_points)
        y = np.array(self.labels)
        self.classes = np.unique(y)
        if len(self.classes) >= 2:
            self.classifier = LogisticRegression(
                random_state=0, C=0.316, max_iter=5000
            ).fit(X, y)
        else:
            self.classifier = None
            if len(self.classes) == 0:
                print("Warning: no classes found in node")
                self.classes = [max(set(self.labels), key=self.labels.count)]

        self.new_data_points = 0

    def train_classifier_essentials_pytorch(self):
        X = np.array(self.data_points)
        y = np.array(self.labels)
        self.classes = np.unique(y)
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}
        def convert_class_to_label(idx):
            return self.class_to_label[idx]
        self.class_to_label_map_func = np.vectorize(convert_class_to_label)
        self.label_to_class = {i: c for c, i in self.class_to_label.items()}
        def convert_label_to_class(idx):
            return self.label_to_class[idx]
        self.label_to_class_map_func = np.vectorize(convert_label_to_class)

        if len(self.classes) >= 2:
            in_dim = self.data_points[0].shape[0]
            device = self.args.device
            self.classifier = PytorchClassifierAsSklearnClassifier(in_dim, len(self.classes), self.label_to_class).to(device)
            
            # TODO: we just hard code the AdamW optimizer here but a better way is to use the one in the args.
            # lr is 0.001, wd is 0.1
            optimizer = torch.optim.AdamW(
                self.classifier.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )

            labels = [
                self.class_to_label_map_func(cls.item()).item()
                for cls in y
            ]
            # Convert labels to tensor
            labels = torch.tensor(labels)
            dl = DataLoader(
                TensorDataset(torch.from_numpy(X), labels),
                batch_size=self.args.batch_size,
                shuffle=True,
            )
            print("Training the linear classifier...")
            correct = 0
            total = 0
            for epoch in range(self.args.n_epochs):
                for batch_idx, (features, labels) in enumerate(dl):
                    features, labels = features.to(self.args.device), labels.to(self.args.device)
                    features = features.float()
                    features.requires_grad_(
                        True
                    )  # Enable gradient computation for features
                    labels = labels.long()
                    optimizer.zero_grad()
                    outputs = self.classifier(features)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    correct += (outputs.argmax(dim=1) == labels).sum().item()
                    total += len(labels)
                    loss.backward()
                    optimizer.step()
                    if batch_idx % 100 == 0:
                        print(
                            "Epoch {}, batch {}, loss {}, accuracy {}".format(
                                epoch, batch_idx, loss.item(), correct / total
                            )
                        )
                        correct = 0
                        total = 0
            print("Finished training the linear classifier")
        else:
            self.classifier = None
            if len(self.classes) == 0:
                print("Warning: no classes found in node")
                self.classes = [max(set(self.labels), key=self.labels.count)]

        self.new_data_points = 0

    def train_classifier(self):
        if self.new:
            if self.args.use_pytorch_linear_for_tree_probe:
                self.train_classifier_essentials_pytorch()
            else:
                self.train_classifier_essentials()
            self.new = False
        else:
            if (
                self.is_leaf
                and len(self.data_points) > 0
                and self.new_data_points >= self.min_samples_for_update
            ):
                if self.args.use_pytorch_linear_for_tree_probe:
                    self.train_classifier_essentials_pytorch()
                else:
                    self.train_classifier_essentials()

    def predict(self, data_point):
        if self.classifier is not None:
            if self.args.use_pytorch_linear_for_tree_probe:
                data_point = torch.from_numpy(data_point).float().to(self.args.device)
                if len(data_point.shape) == 1:
                    data_point = data_point.unsqueeze(0)
                with torch.no_grad():
                    pred = torch.argmax(self.classifier(data_point)).item()
                return self.label_to_class_map_func(pred)
            else:
                return self.classifier.predict([data_point])[0]
        assert len(self.classes) > 0
        return self.classes[0]


class SimpleTree:
    def __init__(self, max_instances=50, min_samples_for_update=10, args=None):
        self.root = SimpleCFNode(
            max_instances, is_leaf=True, min_samples_for_update=min_samples_for_update, args=args
        )
        self.max_instances = max_instances
        self.overall_node_index = 0
        self.min_samples_for_update = min_samples_for_update
        self.args = args

    def _find_closest_node(self, node, data_point):
        if node.is_leaf:
            return node
        else:
            min_distance = float("inf")
            closest_child = None
            for child in node.children:
                distance = np.linalg.norm(child.centroid - data_point)
                if distance < min_distance:
                    min_distance = distance
                    closest_child = child
            return self._find_closest_node(closest_child, data_point)

    def insert(self, data_point, label):
        # Find the closest node and we ensure the closest node is a leaf node
        closest_node = self._find_closest_node(self.root, data_point)
        closest_node.data_points.append(data_point)
        closest_node.labels.append(label)
        closest_node.update_centroid(data_point)
        closest_node.new_data_points += 1

        if len(closest_node.data_points) > self.max_instances:
            self._split_node(closest_node)

    def fit(self, data, labels):
        for point, label in tqdm(zip(data, labels)):
            self.insert(point, label)

    def _split_node(self, node):
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(node.data_points)

        # Create two new child nodes
        self.overall_node_index += 1
        child1 = SimpleCFNode(
            self.max_instances,
            min_samples_for_update=self.min_samples_for_update,
            node_index=self.overall_node_index,
            args=self.args,
        )
        self.overall_node_index += 1
        child2 = SimpleCFNode(
            self.max_instances,
            min_samples_for_update=self.min_samples_for_update,
            node_index=self.overall_node_index,
            args=self.args,
        )

        for i, label in enumerate(kmeans.labels_):
            if label == 0:
                child1.data_points.append(node.data_points[i])
                child1.labels.append(node.labels[i])
                # since we are splitting the node by creating two new nodes, we need to update the new_data_points count for each child
                child1.new_data_points += 1
            else:
                child2.data_points.append(node.data_points[i])
                child2.labels.append(node.labels[i])
                child2.new_data_points += 1

        # Update the centroids of the child nodes
        child1.centroid = kmeans.cluster_centers_[0]
        child2.centroid = kmeans.cluster_centers_[1]

        node.is_leaf = False
        node.children = [child1, child2]
        node.data_points = []
        node.labels = []
        node.new_data_points = 0

    def train_classifiers(self):
        leaf_nodes = self.get_leaf_nodes(self.root)
        for leaf in tqdm(leaf_nodes):
            leaf.train_classifier()

    def get_classifiers(self):
        leaf_nodes = self.get_leaf_nodes(self.root)
        return (
            [leaf.classifier for leaf in leaf_nodes],
            [leaf.classes for leaf in leaf_nodes],
            [leaf.centroid for leaf in leaf_nodes],
        )

    def get_leaf_nodes(self, node):
        if node.is_leaf:
            return [node]
        else:
            leaf_nodes = []
            for child in node.children:
                leaf_nodes.extend(self.get_leaf_nodes(child))
            return leaf_nodes

    def predict(self, data_point):
        closest_node = self._find_closest_node(self.root, data_point)
        return closest_node.predict(data_point)
