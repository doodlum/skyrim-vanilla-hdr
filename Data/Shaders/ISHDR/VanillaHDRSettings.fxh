//--------------------------------------------------------------------------------
//SETTINGS
//--------------------------------------------------------------------------------

#define SETTINGS_VERSION 1

//Changes the way HDR data gets tonemapped
#define TONEMAPPING_METHOD  4	//[1-4]
// 1 = Tonemap every color channel for itself
// 2 = Tonemap peak brightness
// 3 = Tonemap average luma
// 4 = Frostbyte Hue Preservation


//Enables crosstalk between color channels
// -> Desaturates colors as they approach pure white
#define ENABLE_CROSSTALK	0	//[0-1]


//Enables bloom mixing options
#define EBM_ENABLE	1	//[0-1]


// Reduce parameters based on how dark it is
#define ADAPTIVE_SATURATION_ENABLE	1	//[0-1]
#define ADAPTIVE_CONTRAST_ENABLE	1	//[0-1]

// >>>>>>EXTENDED BLOOM MIXING<<<<<< //

#define SETTING_UIB_Saturation 		0.3
#define SETTING_UIB_BloomIntensity 	1.0
#define SETTING_UIB_BloomTint 		float3(1, 1, 1)
#define SETTING_UIB_Contrast 		0.0

// >>>>>>HDR COLOR GRADING<<<<<< //

#define SETTING_UIHCG_Exposure 0.0
#define SETTING_UIHCG_Contrast 1.39
#define SETTING_UIHCG_Saturation 1.0
#define SETTING_UIHCG_ConMiddleGrey 0.43
#define SETTING_UIHCG_Colorbalance float3(1, 1, 1)

// >>>>>>AGIS<<<<<< //

#define SETTING_UIAGIS_SatMin 	0.89
#define SETTING_UIAGIS_SatMax 	2.0
#define SETTING_UIAGIS_SatMix	1.0
#define SETTING_UIAGIS_SatDef	1.0

#define SETTING_UIAGIS_ConMin 	0.9
#define SETTING_UIAGIS_ConMax 	1.25
#define SETTING_UIAGIS_ConMix	1.0
#define SETTING_UIAGIS_ConDef	1.25

#define SETTING_UIAGIS_BrigMin 	0.5
#define SETTING_UIAGIS_BrigMax 	4.0
#define SETTING_UIAGIS_BrigMix 	1.0
#define SETTING_UIAGIS_BrigDef	1.0

#define SETTING_UIAGIS_TintMax 	1.0
#define SETTING_UIAGIS_FadeMax 	1.0

// >>>>>>TONEMAPPER<<<<<< //

#define SETTING_UI_AdaptationMin   	0.10
#define SETTING_UI_AdaptationMax   	0.20
#define SETTING_UI_AdaptationMix   	1.0

// Crosstalk
#define SETTING_UICT_Saturation  	1.0
#define SETTING_UICT_Weight			1.0

// Frostbyte Hue Preservation
#define SETTING_UIFB_Desaturation 	0.7
#define SETTING_UIFB_HueShift 		0.4
#define SETTING_UIFB_Resaturation 	0.0

// Uncharted 2
#define SETTING_UITM_ExposureBias     	1.04  
#define SETTING_UITM_ShoulderStrength 	0.215
#define SETTING_UITM_LinearStrength 	0.255
#define SETTING_UITM_LinearAngle 		0.05
#define SETTING_UITM_ToeStrength 		0.025
#define SETTING_UITM_ToeNumerator 		0.015
#define SETTING_UITM_ToeDenominator 	0.335
#define SETTING_UITM_LinearWhite 		11.2
