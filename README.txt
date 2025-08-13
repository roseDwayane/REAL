py3.8 

**cfg6路徑

1. SimSiam
	EEGLab_Tools_t_SNE: EC
	EEGLab_Tools_t_SNE2: EC&EO
2. Coteaching+
	3 Round:
	EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502
	EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P2
	EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P3
3. CV
	EEGLab_CoT+_FC1_BLA__D512_TC16V3_IMB2__C502_P3
4. Discussion
	EEGLab_LPSO_PT(Adma)__IMB__Revs_BL
		change wl 
		ex. df_wl = df_info[(~df_info['Black_List']) | (df_info['label'] == 1)]
			