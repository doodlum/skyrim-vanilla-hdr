//----------------------------------------------------------------------------------------------//
//																								//
//						enbeffect.fx file by LonelyKitsuune aka Skratzer						//
//						  for Skyrim SE ENB (DirectX 11 Shader Model 5)							//
//																								//
//			   Copyright (c) 2018-2020 LonelyKitsuune / T.Thanner - CC BY-NC-ND 4.0				//
//																								//
//-------------------------------------------CREDITS--------------------------------------------//
//																								//
//								 Boris Vorontsov for ENBSeries									//
//						   Further credits above the respective shaders							//
//																								//
//------------------------------------------THANKS TO-------------------------------------------//
//																								//
//						Timothy Lottes for his VDR Tonemapper presentation						//
//							  John Hable for his filmic worlds blog								//
//							 kingeric1992 and prod80 for inspiration							//
//																								//
//----------------------------------------------------------------------------------------------//
//								==================================								//
//								//     Silent Horizons ENB		//								//
//								//								//								//
//								//		by LonelyKitsuune		//								//
//								==================================								//
//----------------------------------------------------------------------------------------------//

#ifdef SHADERTOOLS
#include "../ISHDR/VanillaHDRSettings.fxh"
#else
#include "VanillaHDRSettings.fxh"
#endif

//----------------------------------------------------------------------------------------------//

float4 Timer;								
float4 ScreenSize;		
float  ENightDayFactor;	
float  EInteriorFactor;	
float  FieldOfView;		
float4 Weather;		
float4 TimeOfDay1;		
float4 TimeOfDay2;					
float4 tempF1;
float4 tempF2;
float4 tempF3;
float4 tempInfo1;
float4 tempInfo2;
SamplerState Linear_Sampler;
#ifdef SHADERTOOLS
#include "../ISHDR/enbHelper_Common.fxh"
#else
#include "enbHelper_Common.fxh"
#endif

#undef DNI_SEPARATION
#undef TODIE_SEPARATION
#define DNI_SEPARATION(x) SETTING_##x
#define TODIE_SEPARATION(x) SETTING_##x

#undef LUM_709
#undef DELTA
#define LUM_709	float3(0.212500006,0.715399981,0.0720999986)
#define DELTA 9.99999975e-06

// Vanilla post-processing based off of work by kingeric1992, aers and nukem
// http://enbseries.enbdev.com/forum/viewtopic.php?f=7&t=5278
// Adapted by doodlez

Texture2D<float4> TextureAdaptation : register(t2);

Texture2D<float4> TextureColor : register(t1);

Texture2D<float4> TextureBloom : register(t0);

SamplerState TextureAdaptationSampler : register(s2);

SamplerState TextureColorSampler : register(s1);

SamplerState TextureBloomSampler : register(s0);

cbuffer cb2 : register(b2)
{
#ifdef FADE
	float4 Params01[6];
#else
	float4 Params01[5];
#endif
}

// Shared PerFrame buffer
cbuffer PerFrame : register(b12) {
#if !defined(VR)
  row_major float4x4 ViewMatrix : packoffset(c0);
  row_major float4x4 ProjMatrix : packoffset(c4);
  row_major float4x4 ViewProjMatrix : packoffset(c8);
  row_major float4x4 ViewProjMatrixUnjittered : packoffset(c12);
  row_major float4x4 PreviousViewProjMatrixUnjittered : packoffset(c16);
  row_major float4x4 InvProjMatrixUnjittered : packoffset(c20);
  row_major float4x4 ProjMatrixUnjittered : packoffset(c24);
  row_major float4x4 InvViewMatrix : packoffset(c28);
  row_major float4x4 InvViewProjMatrix : packoffset(c32);
  row_major float4x4 InvProjMatrix : packoffset(c36);
  float4 CurrentPosAdjust : packoffset(c40);
  float4 PreviousPosAdjust : packoffset(c41);
  // notes: FirstPersonY seems 1.0 regardless of third/first person, could be LE
  // legacy stuff
  float4 GammaInvX_FirstPersonY_AlphaPassZ_CreationKitW : packoffset(c42);
  float4 DynamicRes_WidthX_HeightY_PreviousWidthZ_PreviousHeightW
      : packoffset(c43);
  float4 DynamicRes_InvWidthX_InvHeightY_WidthClampZ_HeightClampW
      : packoffset(c44);
#else
  row_major float4x4 ViewMatrix[2] : packoffset(c0);
  row_major float4x4 ProjMatrix[2] : packoffset(c8);
  row_major float4x4 ViewProjMatrix[2] : packoffset(c16);
  row_major float4x4 ViewProjMatrixUnjittered[2] : packoffset(c24);
  row_major float4x4 PreviousViewProjMatrixUnjittered[2] : packoffset(c32);
  row_major float4x4 InvProjMatrixUnjittered[2] : packoffset(c40);
  row_major float4x4 ProjMatrixUnjittered[2] : packoffset(c48);
  row_major float4x4 InvViewMatrix[2] : packoffset(c56);
  row_major float4x4 InvViewProjMatrix[2] : packoffset(c64);
  row_major float4x4 InvProjMatrix[2] : packoffset(c72);
  float4 CurrentPosAdjust[2] : packoffset(c80);
  float4 PreviousPosAdjust[2] : packoffset(c82);
  // notes: FirstPersonY seems 1.0 regardless of third/first person, could be LE
  // legacy stuff
  float4 GammaInvX_FirstPersonY_AlphaPassZ_CreationKitW : packoffset(c84);
  float4 DynamicRes_WidthX_HeightY_PreviousWidthZ_PreviousHeightW
      : packoffset(c85);
  float4 DynamicRes_InvWidthX_InvHeightY_WidthClampZ_HeightClampW
      : packoffset(c86);
#endif
}

struct VS_OUTPUT
{
	float4 Position : SV_POSITION0;
	float3 TexCoord : TEXCOORD0;
};

typedef VS_OUTPUT PS_INPUT;

struct PS_OUTPUT
{
	float4 Color : SV_Target0;
};

float3 Vanilla(PS_INPUT input)
{
	float2 scaleduv = clamp(0.0, float2(DynamicRes_InvWidthX_InvHeightY_WidthClampZ_HeightClampW.z, DynamicRes_WidthX_HeightY_PreviousWidthZ_PreviousHeightW.y), DynamicRes_WidthX_HeightY_PreviousWidthZ_PreviousHeightW.xy * input.TexCoord.xy);

	float4 color = TextureColor.Sample(TextureColorSampler, scaleduv).rgba;

	bool scaleBloom = (0.5 <= Params01[0].x);
	float3 bloom = TextureBloom.Sample(TextureBloomSampler, scaleBloom ? scaleduv : input.TexCoord.xy).rgb;

	float2 middlegray = TextureAdaptation.Sample(TextureAdaptationSampler, input.TexCoord.xy).xy;

	bool useFilmic = (0.5 < Params01[2].z);
	float WhiteFactor = Params01[2].y;

	float original_lum = max(dot(LUM_709, color.rgb), DELTA);
	float lum_scaled = original_lum * middlegray.y / middlegray.x;
	float lum_filmic = max(lum_scaled - 0.004, 0.0);
	lum_filmic = lum_filmic * (lum_filmic * 6.2 + 0.5) / (lum_filmic * (lum_filmic * 6.2 + 1.7) + 0.06);
	lum_filmic = pow(lum_filmic, 2.2);     // de-gamma-correction for gamma-corrected tonemapper
	lum_filmic = lum_filmic * WhiteFactor; // linear scale
	float lum_reinhard = lum_scaled * (lum_scaled * WhiteFactor + 1.0) / (lum_scaled + 1.0);
	float lum_mapped = useFilmic ? lum_filmic : lum_reinhard;
	color.rgb = color.rgb * lum_mapped / original_lum;

	float bloomfactor = Params01[2].x;
	color.rgb = color.rgb + bloom.rgb * saturate(bloomfactor - lum_mapped);

	float   saturation = Params01[3].x;    // 0 == gray scale
	float   contrast = Params01[3].z;    // 0 == no contrast
	float   brightness = Params01[3].w;    // intensity
	float3  tint_color = Params01[4].rgb;  // tint color
	float   tint_weight = Params01[4].w;    // 0 == no tint
	color.a = dot(color.rgb, LUM_709);
	color.rgb = lerp(color.a, color.rgb, saturation);
	color.rgb = lerp(color.rgb, tint_color * color.a, tint_weight);
	color.rgb = lerp(middlegray.x, color.rgb * brightness, contrast);
	color.rgb = saturate(color.rgb);
#ifdef FADE
	float3  fade = Params01[4].xyz;  // fade current scene to specified color, mostly used in special effects
	float   fade_weight = Params01[4].w;    // 0 == no fade
	color.rgb = lerp(color.rgb, fade, fade_weight);
#endif
	color = saturate(color);
	color = log2(color);
	color = color * GammaInvX_FirstPersonY_AlphaPassZ_CreationKitW.x;
	color = exp2(color);
	return color.rgb;
}




//----------------------------------------------------------------------------------------------//
//											 Functions											//
//																								//
//----------------------------------------------------------------------------------------------//



//Modified Uncharted 2 tonemapper (Original by John Hable)
float  Uncharted2Curve(float A, float B, float C, float D, float E, float F, float  X)
{ return zerolim((X * (A * X + C * B) + D * E) / deltalim(X * (A * X + B) + D * F) - E / F); }

float3 Uncharted2Curve(float A, float B, float C, float D, float E, float F, float3 X)
{ return zerolim((X * (A * X + C * B) + D * E) / deltalim(X * (A * X + B) + D * F) - E / F); }



#define NI nointerpolation

struct TonemapperParams
{
   float  ExposureBias;
   float  ShoulderStrength; //A
   float  LinearStrength;   //B
   float  LinearAngle;      //C
   float  ToeStrength;      //D
   float  ToeNumerator;     //E
   float  ToeDenominator;   //F
   float  LinearWhite;      //W
};

struct ShaderParams
{
NI float  GreyAdapt				: TEXCOORD1;
NI float  UIHCG_Exposure		: TODIE0;
NI float  UIHCG_Contrast		: TODIE1;
NI float  UIHCG_ConMiddleGrey	: TODIE2;
NI float  UIHCG_Saturation		: TODIE3;
NI float3 UIHCG_Colorbalance	: TODIE4;
NI TonemapperParams UITM		: TODIE5;
NI float  UIAGIS_Tint			: AGIS0;
#ifdef FADE
NI float  UIAGIS_Fade			: AGIS1;
#endif

#if EBM_ENABLE
NI float  UIB_Saturation		: BLOOM0;
NI float  UIB_BloomIntensity	: BLOOM1;
NI float3 UIB_BloomTint			: BLOOM2;
NI float  UIB_Contrast			: BLOOM3;
#endif
};


float Tonemap(float Luma, TonemapperParams IN)
{
	Luma *= IN.ExposureBias;
	Luma  = Uncharted2Curve(IN.ShoulderStrength, IN.LinearStrength, IN.LinearAngle,
							IN.ToeStrength, IN.ToeNumerator, IN.ToeDenominator, Luma);
	Luma  = saturate(Luma * IN.LinearWhite);
	return Luma;
}

float3 Tonemap(float3 Color, TonemapperParams IN)
{
	Color *= IN.ExposureBias;
	Color  = Uncharted2Curve(IN.ShoulderStrength, IN.LinearStrength, IN.LinearAngle,
								IN.ToeStrength, IN.ToeNumerator, IN.ToeDenominator, Color);
	Color  = saturate(Color * IN.LinearWhite);
	return Color;
}

// http://enbseries.enbdev.com/forum/viewtopic.php?t=6239

#ifdef SHADERTOOLS
#include "../ISHDR/ictcp_colorspaces.fx"
#else
#include "ictcp_colorspaces.fx"
#endif

float3 FrostbyteTonemap(float3 Color, TonemapperParams IN) 
{ 
  float3 ictcp = rgb2ictcp(Color); 
  float saturation = pow(smoothstep(1.0, 1.0 - SETTING_UIFB_Desaturation, ictcp.x), 1.3); 
  Color = ictcp2rgb(ictcp * float3(1.0, saturation.xx)); 
  float3 perChannel = Tonemap(Color, IN); 
  float peak = max(Color.x, max(Color.y, Color.z)); 
  Color *= rcp(peak + 1e-6); 
  Color *= Tonemap(peak, IN); ; 
  Color = lerp(Color, perChannel, SETTING_UIFB_HueShift); 
  Color = rgb2ictcp(Color); 
  float saturationBoost = SETTING_UIFB_Resaturation * smoothstep(1.0, 0.5, ictcp.x); 
  Color.yz = lerp(Color.yz, ictcp.yz * Color.x / max(1e-3, ictcp.x), saturationBoost); 
  Color = ictcp2rgb(Color); 
  return Color; 
}



// TRI-DITHERING FUNCTION by SANDWICH-MAKER

float rand11(float x) { return frac(x * 0.024390243); }
float permute(float x) { return ((34.0 * x + 1.0) * x) % 289.0; }

#define remap(v, a, b) (((v) - (a)) / ((b) - (a)))
#define BIT_DEPTH 10

float rand21(float2 uv)
{
	float2 noise = frac(sin(dot(uv, float2(12.9898, 78.233) * 2.0)) * 43758.5453);
	return (noise.x + noise.y) * 0.5;
}

float3 triDither(float3 color, float2 uv, float timer)
{
	static const float bitstep = pow(2.0, BIT_DEPTH) - 1.0;
	static const float lsb = 1.0 / bitstep;
	static const float lobit = 0.5 / bitstep;
	static const float hibit = (bitstep - 0.5) / bitstep;

	float3 m = float3(uv, rand21(uv + timer)) + 1.0;
	float h = permute(permute(permute(m.x) + m.y) + m.z);

	float3 noise1, noise2;
	noise1.x = rand11(h); h = permute(h);
	noise2.x = rand11(h); h = permute(h);
	noise1.y = rand11(h); h = permute(h);
	noise2.y = rand11(h); h = permute(h);
	noise1.z = rand11(h); h = permute(h);
	noise2.z = rand11(h);

	float3 lo = saturate(remap(color.xyz, 0.0, lobit));
	float3 hi = saturate(remap(color.xyz, 1.0, hibit));
	float3 uni = noise1 - 0.5;
	float3 tri = noise1 - noise2;
	return float3(
		lerp(uni.x, tri.x, min(lo.x, hi.x)),
		lerp(uni.y, tri.y, min(lo.y, hi.y)),
		lerp(uni.z, tri.z, min(lo.z, hi.z))) * lsb;
}


PS_OUTPUT main(PS_INPUT input)
{
	//------------------------- Shader Parameters Stage ------------------------//

	ShaderParams OUT;

	#if EBM_ENABLE
		OUT.UIB_Saturation	   = DNI_SEPARATION(UIB_Saturation);
		OUT.UIB_BloomIntensity = DNI_SEPARATION(UIB_BloomIntensity);
		OUT.UIB_BloomTint	   = DNI_SEPARATION(UIB_BloomTint);
		OUT.UIB_Contrast	   = DNI_SEPARATION(UIB_Contrast);
	#endif //EBM_ENABLE
	
	OUT.UIHCG_Exposure	 	= exp2(TODIE_SEPARATION(UIHCG_Exposure));
	OUT.UIHCG_Saturation    = TODIE_SEPARATION(UIHCG_Saturation);
	OUT.UIHCG_Contrast		= TODIE_SEPARATION(UIHCG_Contrast);
	OUT.UIHCG_ConMiddleGrey = TODIE_SEPARATION(UIHCG_ConMiddleGrey);
	OUT.UIHCG_Colorbalance  = TODIE_SEPARATION(UIHCG_Colorbalance);
	OUT.UIHCG_Colorbalance  = ColorToChroma(OUT.UIHCG_Colorbalance);
	OUT.GreyAdapt			= clamp(TextureAdaptation.Sample(TextureAdaptationSampler, input.TexCoord.xy).x,
							  TODIE_SEPARATION(UI_AdaptationMin),
							  TODIE_SEPARATION(UI_AdaptationMax));
	
	OUT.UIAGIS_Tint = min(Params01[4].w, DNI_SEPARATION(UIAGIS_TintMax));
	#ifdef FADE
	OUT.UIAGIS_Fade = min(Params01[5].w, DNI_SEPARATION(UIAGIS_FadeMax));
	#endif

	//Mix imagespace deltas into main color grading parameters
	float IS_Saturation = clamp(Params01[3].x, DNI_SEPARATION(UIAGIS_SatMin ),
											   DNI_SEPARATION(UIAGIS_SatMax ));
	float IS_Contrast   = clamp(Params01[3].z, DNI_SEPARATION(UIAGIS_ConMin ),
											   DNI_SEPARATION(UIAGIS_ConMax ));
	float IS_Brightness = clamp(Params01[3].w, DNI_SEPARATION(UIAGIS_BrigMin),
											   DNI_SEPARATION(UIAGIS_BrigMax));
	
	IS_Saturation = lerp(SETTING_UIAGIS_SatDef, IS_Saturation, SETTING_UIAGIS_SatMix);
	IS_Contrast = lerp(SETTING_UIAGIS_ConDef, IS_Contrast, SETTING_UIAGIS_ConMix);
	IS_Brightness = lerp(SETTING_UIAGIS_BrigDef, IS_Brightness, SETTING_UIAGIS_BrigMix);

	OUT.UIHCG_Saturation += IS_Saturation - 1.0;
	OUT.UIHCG_Contrast   += IS_Contrast   - 1.0;
	OUT.UIHCG_Exposure   += IS_Brightness - 1.0;
	
	OUT.UITM.ExposureBias     = TODIE_SEPARATION(UITM_ExposureBias);
	OUT.UITM.ShoulderStrength = TODIE_SEPARATION(UITM_ShoulderStrength);
	OUT.UITM.LinearStrength   = TODIE_SEPARATION(UITM_LinearStrength);
	OUT.UITM.LinearAngle      = TODIE_SEPARATION(UITM_LinearAngle);
	OUT.UITM.ToeStrength      = TODIE_SEPARATION(UITM_ToeStrength);
	OUT.UITM.ToeNumerator     = TODIE_SEPARATION(UITM_ToeNumerator);
	OUT.UITM.ToeDenominator   = TODIE_SEPARATION(UITM_ToeDenominator);
	OUT.UITM.LinearWhite      = TODIE_SEPARATION(UITM_LinearWhite);
	OUT.UITM.LinearWhite      = rcp(deltalim(Uncharted2Curve(OUT.UITM.ShoulderStrength, OUT.UITM.LinearStrength, OUT.UITM.LinearAngle,
								OUT.UITM.ToeStrength, OUT.UITM.ToeNumerator, OUT.UITM.ToeDenominator, OUT.UITM.LinearWhite)));

	#if EBM_ENABLE
		OUT.UIB_BloomTint = ColorToChroma(OUT.UIB_BloomTint);
	#endif

	//------------------------- Pixel Shader Stage ------------------------//

	ShaderParams IN = OUT;

	float2 scaledUV = clamp(0.0, float2(DynamicRes_InvWidthX_InvHeightY_WidthClampZ_HeightClampW.z, DynamicRes_WidthX_HeightY_PreviousWidthZ_PreviousHeightW.y), DynamicRes_WidthX_HeightY_PreviousWidthZ_PreviousHeightW.xy * input.TexCoord.xy);

	float3 Color = TextureColor.Sample(TextureColorSampler, scaledUV.xy);
	
	bool scaleBloom = (0.5 <= Params01[0].x);
	float bloomFactor = Params01[2].x;
	float3 Bloom = TextureBloom.Sample(TextureBloomSampler, (scaleBloom) ? scaledUV.xy : input.TexCoord.xy);

	//------------------------- Apply Game Imagespace and HDR Color Grading ------------------------//
	
	//Imagespace tint and combined saturation
	float Grey  = dot(Color, K_LUM);
	Color = lerp(Color, Params01[4].rgb * Grey, IN.UIAGIS_Tint);
	Color = zerolim(lerp(Grey, Color, IN.UIHCG_Saturation));

	//Combined logarithmic contrast and exposure adjustment
	Color = log2(Color * IN.UIHCG_Exposure + DELTA);
	Color = zerolim(exp2(lerp(IN.UIHCG_ConMiddleGrey, Color, IN.UIHCG_Contrast)) - DELTA);

	//----------------------------------- Adaptation -----------------------------------//
	
	Color /= lerp(1.0, IN.GreyAdapt, SETTING_UI_AdaptationMix);

	//----------------------------------- Bloom Mixing -----------------------------------//

	#if EBM_ENABLE
		Bloom  = zerolim(lerp(dot(Bloom, K_LUM), Bloom, IN.UIB_Saturation));
		Bloom *= IN.UIB_BloomTint;
		Bloom *= IN.UIB_BloomIntensity;
		Bloom  = lerp(Bloom, Bloom * Bloom, IN.UIB_Contrast);
	#endif

	Color += (Bloom  * saturate(bloomFactor));

	//----------------------------------- Selectable Tonemapping -----------------------------------//

	#if TONEMAPPING_METHOD == 1
		Color = Tonemap(Color, IN.UITM);
	
	#elif TONEMAPPING_METHOD == 2
		Grey   = max3(Color);
		Color /= Grey;
		Color *= Tonemap(Grey, IN.UITM);
	
	#elif TONEMAPPING_METHOD == 3
		Grey   = dot(Color, K_LUM);
		Color /= Grey;
		Color *= Tonemap(Grey, IN.UITM);

	#elif TONEMAPPING_METHOD == 4
		Color = FrostbyteTonemap(Color, IN.UITM);
	#endif

	//----------------------------------- Channel Crosstalk -----------------------------------//

	#if ENABLE_CROSSTALK //Channel crosstalk (Timothy Lottes)
		float  MaxColor = max3(Color);
		float3 ColRatio = Color / MaxColor;
		
		ColRatio = lerp(ColRatio, 1.0, MaxColor);
		ColRatio = pow(ColRatio, UICT_Saturation);
		Color	 = lerp(Color, ColRatio * MaxColor, UICT_Weight);
	#endif

	Color = saturate(Color);
	
	//Colorbalance and imagespace fade
	Color *= IN.UIHCG_Colorbalance;

	#ifdef FADE
		Color  = lerp(Color, Params01[5].rgb, IN.UIAGIS_Fade);
	#endif

	Color = log2(Color);
	Color = Color * GammaInvX_FirstPersonY_AlphaPassZ_CreationKitW.x;
	Color = exp2(Color);

	Color += triDither(Color, scaledUV, Grey);
	PS_OUTPUT psout;
	psout.Color.rgb = Color;
	psout.Color.a = 1.0f;
	return psout;
}