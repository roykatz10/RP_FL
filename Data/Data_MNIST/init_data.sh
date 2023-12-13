mnist_subfolders=("Data_10clients_iid_ed/" "Data_10clients_iid_ned/"  "Data_10clients_niid_ed/" "Data_10clients_niid_ned/" "Data_50clients_iid/" "Data_50clients_niid/")

for d in */;
do
    cd $d
    python save_data.py
    cd ..
done

cp "${data_folders[1]}${mnist_subfolders[0]}x_test.pt" "x_test.pt"
cp "${data_folders[1]}${mnist_subfolders[0]}y_test.pt" "y_test.pt"