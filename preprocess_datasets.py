from modules.preprocessing import ABCDataPreprocessor


def main():
    
    dataset = ABCDataPreprocessor.from_directory('./data/raw/nottingham_database')
    dataset.export_to_file("./data/processed/notthingam_database.abc")
    

if __name__ == "__main__":
    main()