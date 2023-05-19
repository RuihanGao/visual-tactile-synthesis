# ours method
python util/compile_eval_metrics_sinskitG.py -n ours -m _sinskitG_baseline_ours


# ## baselines
# # baseline - pix2pix
# python compile_eval_metrics_sinskitG.py -n pix2pix -m _pix2pix_baseline
# # baseline - pix2pixHD
# python compile_eval_metrics_sinskitG.py -n pix2pixHD -m _pix2pixHD_baseline
# # baseline - SPADE
# python compile_eval_metrics_sinskitG.py -n spade -m _spade_baseline


# ## ablations
# # ablation - abl_allGAN
# python compile_eval_metrics_sinskitG.py -n abl_allGAN -m _sinskitG_abl_allGAN
# # ablation - abl_allrec
# python compile_eval_metrics_sinskitG.py -n abl_allrec -m _sinskitG_abl_allrec
# # ablation - abl_G1_GAN
# python compile_eval_metrics_sinskitG.py -n abl_G1_GAN -m _sinskitG_abl_G1_GAN
# # ablation - abl_G2_GAN
# python compile_eval_metrics_sinskitG.py -n abl_G2_GAN -m _sinskitG_abl_G2_GAN
# # ablation - abl_G1_rec
# python compile_eval_metrics_sinskitG.py -n abl_G1_rec -m _sinskitG_abl_G1_rec
# # ablation - abl_G2_rec
# python compile_eval_metrics_sinskitG.py -n abl_G2_rec -m _sinskitG_abl_G2_rec
# # ablation - abl_G1_VAL
# python compile_eval_metrics_sinskitG.py -n abl_G1_VAL -m _sinskitG_abl_G1_VAL