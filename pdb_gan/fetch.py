from pdb_gan.imports import *


class Fetch:
    def get_data(self, url: str, output: list):
        log.info("Fetching data...")
        flag = False
        for line in urllib.request.urlopen(url):
            if line.startswith(b"____"):
                flag = True
                continue

            if flag:
                str_line = str(line, "utf-8")
                output.append(str_line.split(" "))

            if line == b"\n":
                flag = False

        log.info("Found {} PDB entries".format(len(output)))

        return output

    def parse_data(self, data: list, output: dict):
        log.info("Parsing data...")
        tmp = []
        for line in data:
            raw_chars = ["", ",", "A", "\n", ",\n"]
            condition = lambda: any([_ in line for _ in raw_chars])
            while condition() is True:
                for char in raw_chars:
                    try:
                        line.remove(char)
                    except:
                        pass

            try:
                # If new PDB entry
                if len(line[0]) == 4 and len(line) >= 5:
                    # Keep reference to last PDB entry
                    last_record = line[0]
                    tmp.append(last_record.lower())
                    output[last_record] = line[1:]

                elif len(line) == 2 or len(line) == 4:
                    for _ in line:
                        output[last_record].append(_)

            # End of records
            except IndexError:
                pass

        log.info("Merging SP names and records...")
        for key, value in output.items():
            if len(value) > 4:
                tmp_name = value[2::2]
                tmp_entry = value[3::2]
                new_value = value[:2] + ["\n".join(tmp_name)] + ["\n".join(tmp_entry)]
                assert len(new_value) == 4
                output[key] = new_value

        log.info("Generating DataFrame...")
        df = pd.DataFrame.from_dict(output, orient="index", columns=["Method", "Resolution (A)", "SP name", "SP entry"])

        return df

    def download(self, id):
        temp_path = "temp"
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        for angle in ["front", "side", "top"]:
            path = temp_path + "/{}_{}.png".format(id, angle)
            if not os.path.exists(path):
                url = "https://www.ebi.ac.uk/pdbe/static/entry/{}_deposited_chain_{}_image-200x200.png".format(id,
                                                                                                               angle)
                urllib.request.urlretrieve(url, path)
        self.progress.update(task_id=self.task, advance=1)

    def download_snapshots(self, data: pd.DataFrame):
        log.info("Downloading snapshots using {} workers".format(mp.cpu_count()))
        with Progress() as self.progress:
            self.task = self.progress.add_task("[green]Downloading...", total=len(data.index))
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                for id in data.index:
                    executor.submit(self.download, id.lower())

    @staticmethod
    def get_methods(data: pd.DataFrame):
        methods = pd.value_counts(data["Method"])
        console = Console()

        table = Table(title="Experimental Methods", show_header=True, header_style="bold red")
        table.add_column("Method", justify="center")
        table.add_column("Count", justify="right")
        for method, count in zip(methods.index, methods.values):
            table.add_row(method, str(count))

        console.print(table)
