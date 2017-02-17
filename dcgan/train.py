import trainer
import config

cfg = config.Config(filename_queue="dataset/flowers.tfrecords")
t = trainer.Trainer(cfg)

t.fit()
