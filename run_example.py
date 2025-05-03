''' An example program for running the full PIL-Net machine learning pipeline'''

from PILNet.preprocessing import ExtractFeatures, SplitData_NormalizeFeatures
from PILNet.preprocessing import ComputeReferenceMultipoles

from PILNet.training import TrainNetwork
from PILNet.inference import PredictNetwork

def main():
    """
    Main function for running PIL-Net machine learning pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    ExtractFeatures.main(read_filepath="data/dataset/", save_filepath="data/graphs/")
    SplitData_NormalizeFeatures.main(read_filepath="data/graphs/", save_filepath="data/splits/")
    ComputeReferenceMultipoles.main(read_filepath="data/splits/", save_filepath="data/splits/")
    TrainNetwork.main(read_filepath="data/splits/", save_filepath="saved_models/trained_models/")
    PredictNetwork.main(read_filepath_splits="data/splits/", read_filepath_esp="data/dataset/", read_filepath_model="saved_models/trained_models/")

if __name__ == "__main__":
    main()

