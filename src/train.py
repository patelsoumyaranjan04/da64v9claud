import argparse
import json
import os
import sys
import uuid
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--dataset",default="mnist",choices=["mnist","fashion_mnist"])
    parser.add_argument("-e","--epochs",type=int,default=20)
    parser.add_argument("-b","--batch_size",type=int,default=64)

    parser.add_argument("-l","--loss",default="cross_entropy",choices=["cross_entropy","mse"])
    parser.add_argument("-o","--optimizer",default="rmsprop",choices=["sgd","momentum","nag","rmsprop"])
    parser.add_argument("-lr","--learning_rate",type=float,default=1e-3)
    parser.add_argument("-wd","--weight_decay",type=float,default=0.0)

    parser.add_argument("-nhl","--num_layers",type=int,default=4)
    parser.add_argument("-sz","--hidden_size",type=int,nargs="+",default=[128,128,128])
    parser.add_argument("-a","--activation",default="relu",choices=["sigmoid","tanh","relu"])
    parser.add_argument("-w_i","--weight_init",default="xavier",choices=["random","xavier"])

    parser.add_argument("-w_p","--wandb_project",default="da6401_assignment1")
    parser.add_argument("--wandb_entity",default=None)
    parser.add_argument("--no_wandb",action="store_true")

    parser.add_argument("--model_save_path",default="best_model.npy")

    return parser.parse_args()


def _normalize_hidden(args):
    size = args.hidden_size
    num_hidden = args.num_layers         # ← fix: hidden layers = num_layers - 1

    if len(size) < num_hidden:
        size = size + [size[-1]] * (num_hidden - len(size))

    if len(size) > num_hidden:
        size = size[:num_hidden]

    args.hidden_size = size


def main():

    args = parse_arguments()

    _normalize_hidden(args)

    print(f"Loading dataset: {args.dataset}")

    X_train,y_train,X_val,y_val,X_test,y_test = load_data(args.dataset)

    run=None

    if not args.no_wandb:

        try:

            import wandb

            run=wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                name=f"train_{args.dataset}",
                id=str(uuid.uuid4()),
                reinit=True
            )

        except Exception as e:
            print("wandb disabled:",e)

    model=NeuralNetwork(args)

    print("Training...")

    best=model.train(
        X_train,y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val,
        y_val=y_val,
        wandb_run=run
    )

    if best is not None:
        model.set_weights(best)

    test=model.evaluate(X_test,y_test)

    print(
        f"\nTest |acc {test['accuracy']:.4f} "
        f"|f1 {test['f1']:.4f} "
        f"|precision {test['precision']:.4f} "
        f"|recall {test['recall']:.4f}"
    )

    if run:
        run.log({"test_acc":test["accuracy"],"test_f1":test["f1"]})

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_path = os.path.join(BASE_DIR, args.model_save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    weights = model.get_weights()
    np.save(save_path, weights)

    with open(os.path.join(BASE_DIR, "best_config.json"),"w") as f:
        json.dump(vars(args),f,indent=2)

    print("Model saved:",save_path)

    if run:
        run.finish()


if __name__=="__main__":
    main()