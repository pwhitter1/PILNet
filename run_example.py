''' An example program for running the full PIL-Net pipeline'''

from src.preprocessing import ExtractFeatures, SplitData_NormalizeFeatures
from src.preprocessing import ComputeReferenceMultipoles

from src.training import TrainNetwork
from src.inference import PredictNetwork



def main():

    ExtractFeatures.main(read_filepath="src/data/dataset/", save_filepath="src/data/graphs/")
    SplitData_NormalizeFeatures.main(read_filepath="src/data/graphs/", save_filepath="src/data/splits/")

    ComputeReferenceMultipoles.main(read_filepath="src/data/splits/", save_filepath="src/data/splits/")

    TrainNetwork.main(read_filepath="src/data/splits/", save_filepath="src/models/")
    PredictNetwork.main(read_filepath_splits="src/data/splits/", read_filepath_model="src/models/")

if __name__ == "__main__":
    main()

