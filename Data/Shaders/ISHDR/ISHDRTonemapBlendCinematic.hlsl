// Based off of work by kingeric1992, aers and nukem
// http://enbseries.enbdev.com/forum/viewtopic.php?f=7&t=5278
// Adapted by doodlez, some tidbits from L00ping

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
cbuffer PerFrame : register(b12)
{
	row_major float4x4  ViewMatrix                                                  : packoffset(c0);
	row_major float4x4  ProjMatrix                                                  : packoffset(c4);
	row_major float4x4  ViewProjMatrix                                              : packoffset(c8);
	row_major float4x4  ViewProjMatrixUnjittered                                    : packoffset(c12);
	row_major float4x4  PreviousViewProjMatrixUnjittered                            : packoffset(c16);
	row_major float4x4  InvProjMatrixUnjittered                                     : packoffset(c20);
	row_major float4x4  ProjMatrixUnjittered                                        : packoffset(c24);
	row_major float4x4  InvViewMatrix                                               : packoffset(c28);
	row_major float4x4  InvViewProjMatrix                                           : packoffset(c32);
	row_major float4x4  InvProjMatrix                                               : packoffset(c36);
	float4              CurrentPosAdjust                                            : packoffset(c40);
	float4              PreviousPosAdjust                                           : packoffset(c41);
	// notes: FirstPersonY seems 1.0 regardless of third/first person, could be LE legacy stuff
	float4              GammaInvX_FirstPersonY_AlphaPassZ_CreationKitW              : packoffset(c42);
	float4              DynamicRes_WidthX_HeightY_PreviousWidthZ_PreviousHeightW    : packoffset(c43);
	float4              DynamicRes_InvWidthX_InvHeightY_WidthClampZ_HeightClampW    : packoffset(c44);
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

#define DELTA 9.99999975e-06
#define LUM_709 float3(0.212500006,0.715399981,0.0720999986)

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

// http://filmicworlds.com/blog/filmic-tonemapping-operators/

float3 Uncharted2Tonemap(float3 x) {
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

float3 Uncharted2(float3 color, float exposureBias, float W) {
	float3 curr = Uncharted2Tonemap(exposureBias * color);
	float3 whiteScale = 1.0 / Uncharted2Tonemap(float3(W, W, W));
	return curr * whiteScale;
}

float Luminance(float3 linearRgb)
{
	return dot(linearRgb, float3(0.2126729, 0.7151522, 0.0721750));
}

float3 Saturation(float3 c, float sat)
{
	float luma = Luminance(c);
	return luma.xxx + sat.xxx * (c - luma.xxx);
}

float3 Contrast(float3 c, float midpoint, float contrast)
{
	return (c - midpoint) * contrast + midpoint;
}

static const float3x3 LIN_2_LMS_MAT = {
	3.90405e-1, 5.49941e-1, 8.92632e-3,
	7.08416e-2, 9.63172e-1, 1.35775e-3,
	2.31082e-2, 1.28021e-1, 9.36245e-1
};

static const float3x3 LMS_2_LIN_MAT = {
	2.85847e+0, -1.62879e+0, -2.48910e-2,
	-2.10182e-1,  1.15820e+0,  3.24281e-4,
	-4.18120e-2, -1.18169e-1,  1.06867e+0
};

float3 WhiteBalance(float3 c, float3 balance)
{
	float3 lms = mul(LIN_2_LMS_MAT, c);
	lms *= balance;
	return mul(LMS_2_LIN_MAT, lms);
}

//
// Alexa LogC converters (El 1000)
// See http://www.vocas.nl/webfm_send/964
// Max range is ~58.85666
//
struct ParamsLogC
{
	float cut;
	float a, b, c, d, e, f;
};

static const ParamsLogC LogC =
{
	0.011361, // cut
	5.555556, // a
	0.047996, // b
	0.244161, // c
	0.386036, // d
	5.301883, // e
	0.092819  // f
};

float3 LinearToLogC(float3 x)
{
	return LogC.c * log10(LogC.a * x + LogC.b) + LogC.d;
}

float3 LogCToLinear(float3 x)
{
	return (pow(10.0, (x - LogC.d) / LogC.c) - LogC.b) / LogC.a;
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
	PS_OUTPUT psout;

	float2 scaleduv = clamp(0.0, float2(DynamicRes_InvWidthX_InvHeightY_WidthClampZ_HeightClampW.z, DynamicRes_WidthX_HeightY_PreviousWidthZ_PreviousHeightW.y), DynamicRes_WidthX_HeightY_PreviousWidthZ_PreviousHeightW.xy * input.TexCoord.xy);

	float3 color = TextureColor.Sample(TextureColorSampler, scaleduv.xy);

	bool scalebloom = (0.5 <= Params01[0].x);
	float3 bloom = TextureBloom.Sample(TextureBloomSampler, (scalebloom) ? scaleduv.xy : input.TexCoord.xy);
	bloom = lerp(Luminance(bloom), bloom, 0.5f) * 0.5f;

	float2 middlegray = TextureAdaptation.Sample(TextureAdaptationSampler, input.TexCoord.xy).xy;

	float bloomFactor = Params01[2].x;

	color += (bloom * bloomFactor) / (1 + color);

	float   brightness = Params01[3].w;    // intensity
	float   saturation = Params01[3].x;   // 0 == gray scale
	float   contrast = Params01[3].z;    // 0 == no contrast
	float3  tint_color = Params01[4].rgb;  // tint color
	float   tint_weight = Params01[4].a;    // 0 == no tint
	color *= middlegray.y / middlegray.x;
	color *= brightness;
	color /= GammaInvX_FirstPersonY_AlphaPassZ_CreationKitW.x;

	bool useFilmic = (0.5 < Params01[2].z);

	color = LinearToLogC(color);
	color = Contrast(color, 0.5f, contrast);
	color = LogCToLinear(color);

	float whiteFactor = 32.0f / Params01[2].y;
	color = Uncharted2(color, (useFilmic ? 8.0f : 10.0f), whiteFactor);

	color = WhiteBalance(color, float3(1.0f, 1.0f, 1.05f));
	float grey = Luminance(color);
	color = lerp(grey, color, saturation * 1.5);
	color = lerp(color, tint_color * grey, tint_weight);

#ifdef FADE
	float3  fade = Params01[5].rgb;  // fade current scene to specified color, mostly used in special effects
	float   fade_weight = Params01[5].a;    // 0 == no fade
	color. = lerp(color, fade, fade_weight);
#endif

	float tempgray = Luminance(color);

	// Reinhard desaturates shadows, Hable does not
	if (!useFilmic) {
		float4	tempvar;
		tempvar.x = saturate(1.0 - tempgray);
		tempvar.x *= tempvar.x;
		tempvar.x *= tempvar.x;
		color = lerp(color, tempgray, saturate(0.75f - tint_weight) * tempvar.x);
	}

	color += triDither(color, scaleduv, tempgray);
	color = saturate(color);
	color = log2(color);
	color *= 1.5;
	color = exp2(color);
	psout.Color.rgb = color;
	psout.Color.a = 1.0f;
	return psout;
}