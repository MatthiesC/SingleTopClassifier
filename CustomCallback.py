from keras.callbacks import Callback


# When using Dropout layers, the calculation of training loss and validation loss is
# not the same. Thus, if you want to compare both loss histories, it makes much more
# sense to compute training loss and validation loss in the same way. This can only be
# achieved (I guess) by passing the training set as a second validation set during
# training which will then be handled in the same way as the actual validation set. In
# the end, you get a comparable history of training loss and validation loss. Since
# vanilla Keras can only pass a single validation set to the model.fit() function, you
# have to write your own, custom callback which can be used as a history (similar to
# the original History() callback).

# Thanks go to the author of this post:
# https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras


class AdditionalValidationSets(Callback):

    """Use this callback to use additional validation sets during training time."""

    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [3, 4]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(x=validation_data,
                                          y=validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)

            for i, result in enumerate(results):
                if i == 0:
                    valuename = validation_set_name + '_loss'
                else:
                    valuename = validation_set_name + '_' + self.model.metrics[i-1].__name__
                self.history.setdefault(valuename, []).append(result)
