from dltrain import TaskBuilder, SimpleTrainer

builder = TaskBuilder('iris')
builder.base.use_epoch(100).use_batch_size(8).use_device('cuda')
builder.model.use_mlp(4, 3, layers=[16, 64, 16])
builder.optimizer.use_adam()
builder.delineator.use_random_split(builder.dataset.use_iris())
builder.criterion.use_cross_entropy()
builder.forward.use_inject()
builder.inject.use_gradient_acquisition('iris/gradient')
SimpleTrainer().run(builder.build())
