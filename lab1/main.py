from sklearn.datasets import load_iris
import numpy as np
import argparse
import torch

rng = np.random.default_rng(0)

dataset = load_iris()

X = dataset['data'].astype(np.float64)
y = dataset['target']

train_indices = []
test_indices = []

for cls in [0,1,2]:
    cls_indices = np.flatnonzero(y == cls)
    rng.shuffle(cls_indices)

    train_indices.append(cls_indices[:35])
    test_indices.append(cls_indices[35:])

train_indices = np.concatenate(train_indices)
test_indices = np.concatenate(test_indices)

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]


train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0, ddof=0)

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std




class CrossEntropyLoss:
    def compute_loss(self, logits: np.ndarray, y: np.ndarray):
        loss = 0.0
        probs = []

        for row, target in zip(logits, y):
            max_logit = np.max(row)
            shifted_row = row - max_logit

            exp_row = np.exp(shifted_row)
            sum_exp = np.sum(exp_row)

            row_probs = exp_row / sum_exp
            probs.append(row_probs)

            loss += shifted_row[target] - np.log(sum_exp)

        self.probs = np.array(probs)
        self.y = y.copy()

        loss *= -1 / len(y)
        self.loss = loss

        return loss
    
    def compute_grad(self, buggy=False):
        grad = self.probs.copy()
        grad[np.arange(len(self.y)), self.y] -= 1
        if not buggy:
            grad /= len(self.y)
        self.grad = grad

        return grad


class OwnMLP:
    def __init__(self):
        init_rng = np.random.default_rng(0)

        self.W_1 = init_rng.normal(loc=0.0, scale=np.sqrt(0.5), size=(4,8))
        self.W_2 = init_rng.normal(loc=0.0, scale=np.sqrt(2 / 11), size=(8,3))

        self.b_1 = np.zeros(8, dtype=np.float64)
        self.b_2 = np.zeros(3, dtype=np.float64)


    def forward(self, X):
        Z_1 = X @ self.W_1 + self.b_1
        A_1 = np.maximum(0, Z_1)
        Z_2 = A_1 @ self.W_2 + self.b_2

        self.saved_values = {"X" : X, "Z_1" : Z_1, "A_1" : A_1, "Z_2" : Z_2}

        return Z_2
        

    def backward(self, grad_Z_2):
        X = self.saved_values["X"]
        Z_1 = self.saved_values["Z_1"]
        A_1 = self.saved_values["A_1"]

        grad_W_2 = A_1.T @ grad_Z_2
        grad_b_2 = grad_Z_2.sum(axis=0)
        grad_A_1 = grad_Z_2 @ self.W_2.T
        grad_Z_1 = grad_A_1 * (Z_1 > 0)
        grad_W_1 = X.T @ grad_Z_1
        grad_b_1 = grad_Z_1.sum(axis=0)

        return {"W_1": grad_W_1, "b_1": grad_b_1,
                "W_2": grad_W_2, "b_2": grad_b_2}


def pytorch_reference(model, X, y):
    reference = torch.nn.Sequential(
        torch.nn.Linear(4, 8, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3, dtype=torch.float64),
    )
    with torch.no_grad():
        reference[0].weight.copy_(torch.from_numpy(model.W_1.T))
        reference[0].bias.copy_(torch.from_numpy(model.b_1))
        reference[2].weight.copy_(torch.from_numpy(model.W_2.T))
        reference[2].bias.copy_(torch.from_numpy(model.b_2))

    logits = reference(torch.from_numpy(X))
    loss = torch.nn.functional.cross_entropy(
        logits, torch.from_numpy(y).long(), reduction="mean")
    loss.backward()
    grad_W_1 = reference[0].weight.grad
    grad_b_1 = reference[0].bias.grad
    grad_W_2 = reference[2].weight.grad
    grad_b_2 = reference[2].bias.grad
    assert grad_W_1 is not None
    assert grad_b_1 is not None
    assert grad_W_2 is not None
    assert grad_b_2 is not None
    grads = {
        "W_1": grad_W_1.numpy().T.copy(),
        "b_1": grad_b_1.numpy().copy(),
        "W_2": grad_W_2.numpy().T.copy(),
        "b_2": grad_b_2.numpy().copy(),
    }
    return loss.item(), grads


def numerical_gradient_check(model, loss_fn, X, y, grads, eps=1e-6):
    results = []
    for name, index in [("W_1", (0, 0)), ("b_1", (0,)),
                        ("W_2", (0, 0)), ("b_2", (0,))]:
        parameter = getattr(model, name)
        original = parameter[index].copy()
        try:
            parameter[index] = original + eps
            loss_plus = loss_fn.compute_loss(model.forward(X), y)
            parameter[index] = original - eps
            loss_minus = loss_fn.compute_loss(model.forward(X), y)
        finally:
            parameter[index] = original

        numerical = (loss_plus - loss_minus) / (2 * eps)
        manual = grads[name][index]
        diff = abs(manual - numerical)
        passed = bool(np.isfinite([loss_plus, loss_minus, manual, numerical, diff]).all()
                      and diff <= 1e-7)
        label = name + "[" + ",".join(map(str, index)) + "]"
        results.append((label, manual, numerical, diff, passed))

  
    loss_fn.compute_loss(model.forward(X), y)
    return results


def compare_values(numpy_value, torch_value, tolerance=1e-12):
    diff = np.max(np.abs(numpy_value - torch_value))
    passed = bool(np.isfinite(numpy_value).all()
                  and np.isfinite(torch_value).all()
                  and np.isfinite(diff) and diff <= tolerance)
    return diff, passed


def main():
    parser = argparse.ArgumentParser(description="Manual backpropagation checks on Iris")
    parser.add_argument("--bug-experiment", action="store_true",
                        help="omit division by N only in the logits gradient")
    args = parser.parse_args()

    model = OwnMLP()
    loss_fn = CrossEntropyLoss()
    numpy_loss = loss_fn.compute_loss(model.forward(X_train), y_train)
    correct_grads = model.backward(loss_fn.compute_grad())
    grads = model.backward(loss_fn.compute_grad(buggy=args.bug_experiment))
    torch_loss, torch_grads = pytorch_reference(model, X_train, y_train)

    print("Mode:", "intentional bug (no 1/N)" if args.bug_experiment else "correct")
    print(f"NumPy loss:   {numpy_loss:.17g}")
    print(f"PyTorch loss: {torch_loss:.17g}")
    print("NumPy/PyTorch max absolute differences (tolerance 1e-12):")
    comparisons = [compare_values(numpy_loss, torch_loss)]
    comparisons.extend(compare_values(grads[name], torch_grads[name])
                       for name in grads)
    for label, (diff, passed) in zip(["Loss"] + list(grads), comparisons):
        print(f"  {label:5s} {diff:.12e}  {'PASS' if passed else 'FAIL'}")

    numerical_results = numerical_gradient_check(model, loss_fn, X_train, y_train, grads)
    print("Central differences (epsilon 1e-6, tolerance 1e-7):")
    print(f"{'Parameter':12s} {'Manual':>20s} {'Numerical':>20s} {'Abs diff':>18s}  Result")
    for label, manual, numerical, diff, passed in numerical_results:
        print(f"{label:12s} {manual:20.12e} {numerical:20.12e} {diff:18.12e}  "
              f"{'PASS' if passed else 'FAIL'}")

    if args.bug_experiment:
        print(f"Gradient scaling (buggy/correct, expected N={len(y_train)}):")
        for name in grads:
            nonzero = correct_grads[name] != 0
            ratios = grads[name][nonzero] / correct_grads[name][nonzero]
            zeros_preserved = bool((grads[name][~nonzero] == 0).all())
            print(f"  {name}: ratio [{ratios.min():.12g}, {ratios.max():.12g}], "
                  f"zero components {np.count_nonzero(~nonzero)}, preserved={zeros_preserved}")
        detected = (comparisons[0][1]
                    and all(not passed for _, passed in comparisons[1:])
                    and all(not row[-1] for row in numerical_results))
        print("Intentional bug detected:", "PASS" if detected else "FAIL")
        return 0 if detected else 1

    passed = all(result[1] for result in comparisons) and all(row[-1] for row in numerical_results)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


   
    
         



            
