''' An example program for running the full PIL-Net pipeline'''

# from src.preprocessing import ExtractFeatures, SplitData_NormalizeFeatures
# from src.training import TrainNetwork
# from src.inference import PredictNetwork

from src.preprocessing import ComputeReferenceMultipoles

def main():

    # ExtractFeatures.main(read_filepath="src/datasets/", save_filepath="src/graphs/")
    # SplitData_NormalizeFeatures.main(read_filepath="src/graphs/", save_filepath="src/splits/")
    # ComputeReferenceMultipoles.main(read_filepath="src/splits/", save_filepath="src/splits/")

    # TrainNetwork.main(read_filepath="src/splits/", save_filepath="src/models/")
    # PredictNetwork.main(read_filepath_splits="src/splits/", read_filepath_model="src/models/")


    # ExtractFeatures.main(read_filepath="/scratch/gilbreth/cwhitter/data/QMDatasetForAtomicMultipoles_2022/", save_filepath="/scratch/gilbreth/cwhitter/data/QMDatasetForAtomicMultipoles_2022/")
    # SplitData_NormalizeFeatures.main(read_filepath="/scratch/gilbreth/cwhitter/data/QMDatasetForAtomicMultipoles_2022/", save_filepath="/scratch/gilbreth/cwhitter/data/QMDatasetForAtomicMultipoles_2022/")
    ComputeReferenceMultipoles.main(read_filepath="/scratch/gilbreth/cwhitter/data/QMDatasetForAtomicMultipoles_2022/", save_filepath="/scratch/gilbreth/cwhitter/data/QMDatasetForAtomicMultipoles_2022/")

    # TrainNetwork.main(read_filepath="./", save_filepath="src/models/")
    # PredictNetwork.main(read_filepath_splits="./", read_filepath_model="src/models/")

if __name__ == "__main__":
    main()

