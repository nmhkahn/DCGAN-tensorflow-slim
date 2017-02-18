import scipy.misc
import numpy as np

import ops
import config
import trainer

def main():
    cfg = config.Config(filename_queue="dataset/flowers.tfrecords")
    t = trainer.Trainer(cfg)

    _, im = t.sample(100)
    for i in range(100):
        scipy.misc.imsave("example/"+str(i+1)+".jpg", im[i])

if __name__ == "__main__":
    main()
