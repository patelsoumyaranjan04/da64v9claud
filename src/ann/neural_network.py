import numpy as np
from ann.neural_layer import Layer
from ann.objective_functions import LOSS_FN, LOSS_GRAD
from ann.optimizers import OPTIMIZERS
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class NeuralNetwork:

    def __init__(self, args):

        self.args = args
        self.layers = []

        self._construct_network()

        opt_class = OPTIMIZERS[args.optimizer]
        self.optimizer = opt_class(
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        self.optimizer.init_state(self.layers)

    def _construct_network(self):

        a = self.args

        num_layers = getattr(a, "num_layers", 3)
        hidden_size = getattr(a, "hidden_size", [128] * (num_layers - 1))

        hidden_sizes = hidden_size if isinstance(hidden_size, list) else [hidden_size] * (num_layers - 1)

        num_hidden = num_layers 

        if len(hidden_sizes) < num_hidden:
            hidden_sizes = hidden_sizes + [hidden_sizes[-1]] * (num_hidden - len(hidden_sizes))

        elif len(hidden_sizes) > num_hidden:
            hidden_sizes = hidden_sizes[:num_hidden]

        dims = [784] + hidden_sizes + [10]

        activation = getattr(a, "activation", "relu")
        weight_init = getattr(a, "weight_init", "xavier")

        for i in range(len(dims) - 1):

            act = activation if i < len(dims) - 2 else None
            self.layers.append(Layer(dims[i], dims[i + 1], act, weight_init))

    def forward(self, X):

        out = X

        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, y, logits):

        delta = LOSS_GRAD[self.args.loss](logits, y)

        grads_w = []
        grads_b = []

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

            grads_w.insert(0, layer.grad_W)
            grads_b.insert(0, layer.grad_b)

        return grads_w, grads_b

    def update_weights(self):
        self.optimizer.step(self.layers)

    def _batch_iterator(self, X, y, batch):

        n = X.shape[0]

        for start in range(0, n, batch):
            end = start + batch
            yield X[start:end], y[start:end]

    def train(self, X_train, y_train, epochs, batch_size,
              X_val=None, y_val=None, wandb_run=None):

        n = X_train.shape[0]

        best_f1 = -1
        best_weights = None

        for epoch in range(epochs):

            order = np.random.permutation(n)
            X_train = X_train[order]
            y_train = y_train[order]

            total_loss = 0.0

            for Xb, yb in self._batch_iterator(X_train, y_train, batch_size):

                logits = self.forward(Xb)

                loss = LOSS_FN[self.args.loss](logits, yb)
                total_loss += loss * len(yb)

                self.backward(yb, logits)
                self.update_weights()

            epoch_loss = total_loss / n

            train_metrics = self.evaluate(X_train, y_train)

            log = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_acc": train_metrics["accuracy"]
            }

            if X_val is not None:

                val_metrics = self.evaluate(X_val, y_val)

                log.update({
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"]
                })

                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    best_weights = self.get_weights()

            if wandb_run:
                wandb_run.log(log)

            msg = f"Epoch {epoch+1}/{epochs} | loss {epoch_loss:.4f} | train_acc {train_metrics['accuracy']:.4f}"

            if X_val is not None:
                msg += f" | val_acc {log['val_acc']:.4f}"

            print(msg)

        return best_weights

    def evaluate(self, X, y):

        logits = self.forward(X)

        loss = LOSS_FN[self.args.loss](logits, y)

        preds = np.argmax(logits, axis=1)

        metrics = {
            "loss": loss,
            "accuracy": accuracy_score(y, preds),
            "f1": f1_score(y, preds, average="macro", zero_division=0),
            "precision": precision_score(y, preds, average="macro", zero_division=0),
            "recall": recall_score(y, preds, average="macro", zero_division=0),
            "logits": logits
        }

        return metrics

    def get_weights(self):

        data = {}

        for i, layer in enumerate(self.layers):
            data[f"W{i}"] = layer.W.copy()
            data[f"b{i}"] = layer.b.copy()

        return data

    def set_weights(self, weights):
        num_layers = len([k for k in weights if k.startswith("W")])

        # Use W.shape[0] instead of in_features (Layer class doesn't have in_features attr)
        if num_layers != len(self.layers) or weights["W0"].shape[0] != self.layers[0].W.shape[0]:
            self.layers = []
            for i in range(num_layers):
                W = weights[f"W{i}"]
                in_dim = W.shape[0]
                out_dim = W.shape[1]
                act = self.args.activation if i < num_layers - 1 else None
                self.layers.append(Layer(in_dim, out_dim, act, self.args.weight_init))

        for i, layer in enumerate(self.layers):
            layer.W = weights[f"W{i}"].copy()
            layer.b = weights[f"b{i}"].copy()