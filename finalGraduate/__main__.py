
import warnings
from transformers import logging
import datasets

# datasets.logging.set_verbosity_error()
# logging.set_verbosity_error()
# warnings.simplefilter("ignore")
from models.Xlnet_glue import cola_run

if __name__ == '__main__':
    # for i in [None, "dynamic", "static", "staticTr", "staticTrQsin"]:

    for i in [None]:
        # for j in ["mrpc", "rte", "sst2", "cola", "stsb", "qqp", "qnli", "mnli"]:
        for j in ["mrpc"]:

            print(f'\n{j}\n')
            for k in range(1):
                cola_run(quant_type=i, task=j)
            print("\n\n")
