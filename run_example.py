''' An example program for running the full PIL-Net pipeline'''

from src.preprocessing import ExtractFeatures, SplitData_NormalizeFeatures, ComputeReferenceMultipoles
from src.training import TrainNetwork
from src.inference import PredictNetwork

def main():

    ExtractFeatures.main(read_filepath="datasets/", save_filepath="graphs/")
    SplitData_NormalizeFeatures.main(read_filepath="graphs/", save_filepath="splits/")
    ComputeReferenceMultipoles.main(read_filepath="splits/", save_filepath="splits/")

    TrainNetwork.main(read_filepath="splits/", save_filepath="models/")
    PredictNetwork.main(read_filepath_splits="splits/", read_filepath_model="models/")

if __name__ == "__main__":
    main()

