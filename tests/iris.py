from dltrain import TaskBuilder, SimpleTrainer

builder = TaskBuilder('mnist')
builder.base.use_device('cuda')
builder.optimizer.use_adam()
builder.criterion.use_cross_entropy()
builder.evaluation_handler.add_accuracy()
builder.delineator.use_train_eval(builder.dataset.use_mnist('./dataset', True),
                                  builder.dataset.use_mnist('./dataset', False))
SimpleTrainer().run(builder.build())
