from pathlib import Path

if __name__ == '__main__':
    path_root = Path(r"D:\Durability\kinematics\v3d_reports")
    for participant_path in path_root.glob("DUR*"):
        participant_id = participant_path.stem
        for shoe_path in participant_path.glob("*AFT"):
            shoe_condition = shoe_path.stem
            for file in shoe_path.glob("*cmz"):
                stem = file.stem.strip()
                print(participant_id, shoe_condition)
                if stem.startswith("1-"):
                    print("Matched 1-")
                    new_filename = f"{participant_id}_{shoe_condition}_Report_1-5.cmz"
                elif stem.startswith("6-"):
                    print("Matched 6-")
                    new_filename = f"{participant_id}_{shoe_condition}_Report_6-10.cmz"
                elif stem.startswith("11"):
                    print("Matched 11")
                    new_filename = f"{participant_id}_{shoe_condition}_Report_11.cmz"
                else:
                    print("Else clause reached")
                    continue
                file.rename(file.with_name(new_filename))
