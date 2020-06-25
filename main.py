from pdb_gan.imports import *


def main():
    def save(data: pd.DataFrame):
        with open("data\parsed.pkl", "wb") as file:
            pickle.dump(data, file)

    fetch = Fetch()

    path = "data"
    if not os.path.exists(path):
        os.makedirs(path)

    records = fetch.get_data(url="https://www.uniprot.org/docs/pdbtosp.txt", output=[])
    parsed = fetch.parse_data(data=records, output={})

    try:
        with open("data\parsed.pkl", "rb") as file:
            temp = pickle.load(file)

        if len(temp) == len(parsed):
            log.info("Data is already up-to-date")
        else:
            save(data=parsed)

    except FileNotFoundError:
        log.info("Saving data...")
        save(data=parsed)

    fetch.get_methods(data=parsed)
    fetch.download_snapshots(data=parsed)


if __name__ == "__main__":
    main()
