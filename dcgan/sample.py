import scipy.misc
import numpy as np

import ops
import config
import trainer

def main():
    cfg = config.Config(filename_queue="dataset/flowers.tfrecords")
    t = trainer.Trainer(cfg)

    if not os.path.exists(cfg.sampledir):
        os.makedirs(cfg.sampledir)

    _, im = t.sample(100)
    for i in range(100):
        imname = os.path.join(config.sampledir, str(step+1)+".jpg")
        scipy.misc.imsave("example/"+str(i+1)+".jpg", im[i])

if __name__ == "__main__":
    main()
