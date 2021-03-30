#!/bin/bash


while [ $# -ne 0 ]
do
	case "$1" in
		-i|--imagen)
			IMAGEN="$2"
			;;
		-k|--kernel)
		 	KERNEL="$2"
		 	;;
#		 -e|--epochs)
#			EPOCHS=$2
#			;;
	esac
    shift
done



python2 extract_atts_lung.py  -i Train/tr_im$IMAGEN.png  -m  LungMask/tr_lungmask$IMAGEN.png  -c Mask/tr_mask$IMAGEN.png  -r "results/im_"$IMAGEN"_k_"$KERNEL".csv"  -rr "results/im_"$IMAGEN"_k_"$KERNEL"_n.csv"  -s "results/im_"$IMAGEN"_k_"$KERNEL"_pix.csv"  -o "new_images/im_"$IMAGEN".png"  -o2 "new_images/im_"$IMAGEN"_k_"$KERNEL".png" -nk $KERNEL

echo "extract_atts_lung.py DONE" 

#python2 run_somfft.py  -i "results/im_"$IMAGEN"_k_"$KERNEL"_n.csv"  -e $EPOCHS  -m 3  -n 3  -wt "som/im_"$IMAGEN"_k_"$KERNEL"_n_3_n.wt"  -mp "som/im_"$IMAGEN"_k_"$KERNEL"_n_3_n.map"  -bmu x.csv  -c som  -stats "som/im_"$IMAGEN"_k_"$KERNEL"_n_3_n.err"  -vec y.csv  -bs b.csv  -HF 2  -bD 1.0

#echo "run_somfft.py DONE" 

python2 AnomDet_Lung.py  -i "results/im_"$IMAGEN"_k_"$KERNEL"_n.csv" -m if  -th 10  -o "AD/im_"$IMAGEN"_k_"$KERNEL"_n_if_th_10.csv"

echo "AnomDet_Lung.py DONE" 

#python2 new_code_lung_AD.py  -i  "new_images/im_"$IMAGEN".png"  -mp "som/im_"$IMAGEN"_k_"$KERNEL"_n_3_n.map"  -ad "AD/im_"$IMAGEN"_k_"$KERNEL"_n_if_th_10.csv"  -o  "new_images/im_"$IMAGEN"_k_"$KERNEL"_n_3_n_AD_if_th_10.png"  -r "AD/im_"$IMAGEN"_k_"$KERNEL"_n_3_n_AD_if_th_10.csv" -n 5