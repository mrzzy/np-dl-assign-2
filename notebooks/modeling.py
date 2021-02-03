#
# np-dl-assign-2
# Shared Model building and training utilities
#

import os
import mlflow
from git import Repo
from shutil import rmtree
from tempfile import mkdtemp, NamedTemporaryFile

from tensorflow.keras.layers import (
    Activation,
    LeakyReLU,
    ReLU,
    LSTM,
    GRU,
    BatchNormalization,
    LayerNormalization,
    Dense,
)
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import Input, Model

def dense_classifier(
    in_op,
    n_dense_units=128,
    n_classes=10,
    use_batch_norm=True,
    dropout_prob=0.0,
    l2_reg=None,
    dense_activation="relu",
    **activation_params,
):
    """Append a dense classfier block to the given in_op"""
    x = in_op

    # disable hidden dense layer if n_dense_units < 1
    if n_dense_units >= 1:
        x = Dense(
            units=n_dense_units,
            kernel_regularizer=None if l2_reg is None else l2(l=l2_reg),
        )(x)
        if use_batch_norm:
            x = BatchNormalization()(x)

        x = Activation(dense_activation)(x)
    # classifier output layer
    x = Dense(
        units=n_classes,
        activation="softmax",
        kernel_regularizer=None if l2_reg is None else l2(l=l2_reg),
    )(x)
    return x


def rnn_block(
    in_op,
    rnn_cell="lstm",
    n_rnn_units=64,
    rnn_activation="tanh",
    use_layer_norm=True,
    return_sequences=False,
    dropout_prob=0.0,
    l2_reg=None,
    **activation_params,
):
    """Append a RNN layer block to the given in_op"""
    rnn_params = {
        "units": n_rnn_units,
        "activation": rnn_activation,
        "recurrent_dropout": dropout_prob,
        "kernel_regularizer": None if l2_reg is None else l2(l=l2_reg),
        "return_sequences": return_sequences,
    }

    x = in_op
    if rnn_cell == "lstm":
        x = LSTM(**rnn_params)(x)
    elif rnn_cell == "gru":
        x = GRU(**rnn_params)(x)
    else:
        raise NotImplementedError(f"Unsupported RNN cell: {rnn_cell}")

    if use_layer_norm:
        x = LayerNormalization()(x)

    return x


def compile_callbacks(reduce_lr_stuck, reduce_lr_patience, reduce_lr_factor):
    """Compile training callbacks"""
    # compile callbacks
    callbacks = []
    # callback to log tensorboard
    log_dir = mkdtemp()
    tensorboard = TensorBoard(log_dir=log_dir)
    callbacks.append(tensorboard)

    # callback to reduce lr on stuck on plateau
    mlflow.log_param("reduce_lr_stuck", reduce_lr_stuck)
    if reduce_lr_stuck:
        callbacks.append(
            ReduceLROnPlateau(
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                monitor="loss",
            )
        )
        mlflow.log_params(
            {
                "reduce_lr_factor": reduce_lr_factor,
                "reduce_lr_patience": reduce_lr_patience,
            }
        )
    # callback to log to tensorboard during model training
    def on_epoch_end(n_epoch, logs):
        mlflow.log_metrics(logs, step=n_epoch)

    def on_train_end(logs=None):
        # due to this issue https:// pgithub.com/tensorflow/tensorflow/issues/38498
        # logs is always none.
        mlflow.log_artifacts(log_dir, "tensorboard_logs")
        rmtree(log_dir, ignore_errors=True)

    mlflow_log = LambdaCallback(
        on_epoch_end=on_epoch_end,
        on_train_end=on_train_end,
    )
    callbacks.append(mlflow_log)
    return callbacks

def build_model(
    input_shape,
    n_classes,
    n_dense_units=0,
    use_batch_norm=True,
    rnn_cell="lstm",
    n_rnn_layers=3,
    n_rnn_units=64,
    rnn_activation="tanh",
    dense_activation="relu",
    use_layer_norm=True,
    **block_params
):
    in_op = Input(shape=input_shape)
    x = in_op

    for i_layer in range(1, n_rnn_layers + 1):
        x = rnn_block(
            in_op=x,
            rnn_cell=rnn_cell,
            n_rnn_units=n_rnn_units,
            rnn_activation=rnn_activation,
            use_layer_norm=use_layer_norm,
            # return sequences except last rnn layer
            return_sequences=True if i_layer < n_rnn_layers else False,
            **block_params,
        )

    x = dense_classifier(
        in_op=x,
        n_classes=n_classes,
        n_dense_units=n_dense_units,
        use_batch_norm=use_batch_norm,
        dense_activation=dense_activation,
        **block_params,
    )

    return Model(
        inputs=in_op,
        outputs=x,
    )


def train_eval_model(
    train_data,
    test_data,
    git_repo=Repo(search_parent_directories=True),
    tags={},
    build_model_fn=build_model,
    epochs=30,
    lr=1e-3,
    optimizer="adam",
    sgd_momentum=0.9,
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
    ],
    validation_split=0.1,
    img_size=(256, 256),
    reduce_lr_stuck=False,
    reduce_lr_patience=5,
    reduce_lr_factor=0.1,
    batch_size=32,
    run_name=None,
    **build_params,
):
    
    # configure run tags: record version (git ref) information
    tags["git-ref"] = git_repo.head.commit.hexsha
    tags["version"] = git_repo.git.describe("--always")
    mlflow.set_tags(tags)

    # build model
    model = build_model_fn(**build_params)
    mlflow.log_params(build_params)

    # write summary to mlflow
    temp_dir = mkdtemp()
    summary_fpath = os.path.join(temp_dir, "summary.txt")
    with open(summary_fpath, "w") as f:
        model.summary(print_fn=(lambda line: f.write(f"{line}\n")))
    mlflow.log_artifact(summary_fpath)
    rmtree(temp_dir)

    # compile model
    optimizers = {
        "adam": Adam(learning_rate=lr),
        "rmsprop": RMSprop(learning_rate=lr),
        "sgd": SGD(learning_rate=lr, momentum=sgd_momentum),
    }
    model.compile(
        optimizer=optimizers[optimizer],
        loss=loss,
        metrics=metrics,
    )
    mlflow.log_params(
        {
            "optimizer": optimizer,
            "loss": loss,
            "metrics": metrics,
            "learning_rate": lr,
        }
    )
    if optimizer == "sgd":
        mlflow.log_param("sgd_momentum", sgd_momentum)

    # train model
    mlflow.log_param("fit_epochs", epochs)
    mlflow.log_param("validation_split", validation_split)
    callbacks = compile_callbacks(
        reduce_lr_stuck=reduce_lr_stuck,
        reduce_lr_factor=reduce_lr_factor,
        reduce_lr_patience=reduce_lr_patience,
    )
    _ = model.fit(
        *train_data,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
    )
    mlflow.keras.log_model(model, "models")
    # evaluate model
    test_metrics = model.evaluate(*test_data, verbose=2)
    # prefix metrics names with test to indicate computed on test set
    test_metric_names = ["test_" + name for name in model.metrics_names]
    mlflow.log_metrics(dict(zip(test_metric_names, test_metrics)))

    return model