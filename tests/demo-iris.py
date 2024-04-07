from src.dltrain import TaskBuilder, SimpleTrainer, DatasetWizard
from src.dltrain.builder import AdamWBuilder, StepLrBuilder
from src.dltrain.evaluation import Accuracy, ConfusionMatrix
from src.dltrain import Standardize
from argparse import ArgumentParser


def set_options():
    parser = ArgumentParser()

    parser.add_argument('--version', default='1.0.0', type=str, required=False, help='版本信息')
    parser.add_argument('--device', default='cuda', type=str, required=False, help='训练设备')
    parser.add_argument('--epochs', default=30, type=int, required=False, help='训练批次')
    parser.add_argument('--batch_size', default=16, type=int, required=False, help='批数量')
    parser.add_argument('--lr', default=0.01, type=float, required=False, help='学习率')

    return parser.parse_args()


def main(options):
    builder = TaskBuilder()

    # 设置任务名称
    builder.use_task_name('demo_training_iris')

    # 使用随机划分数据集(使用DatasetWizard导入常用数据集)
    builder.use_random_split_dataset(DatasetWizard.use_iris(), 0.8, 0.2)

    # 数据集预处理手段
    builder.add_features_transform(Standardize())

    # 设置模型
    builder.use_mlp(4, 3, activation='relu')

    # 使用config简化参数设置
    builder.use_config(
        device=options.device,
        epochs=options.epochs,
        batch_size=options.batch_size
    )

    # 允许保存checkpoint
    builder.use_save_checkpoint(True)

    # 使用交叉熵损失函数
    builder.use_cross_entry()

    # 使用AdamW优化器
    builder.use_optimizer(AdamWBuilder().set_lr(options.lr))

    # 使用StepLR学习率策略
    builder.use_scheduler(StepLrBuilder().set_step_size(10).set_gamma(0.9))

    # 为训练与测试添加验证手段
    builder.add_train_evaluation_handler('Accuracy', Accuracy())
    builder.add_train_evaluation_handler('ConfusionMatrix', ConfusionMatrix())
    builder.add_eval_evaluation_handler('Accuracy', Accuracy())
    builder.add_eval_evaluation_handler('ConfusionMatrix', ConfusionMatrix())

    trainer = SimpleTrainer()
    trainer.run(builder.build())


if __name__ == '__main__':
    main(set_options())
