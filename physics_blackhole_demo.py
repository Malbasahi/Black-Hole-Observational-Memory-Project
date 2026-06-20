# black_hole_kerr_reference_simulator_science_calibrated_cinematic.py
# Real-time GPU black-hole visualization with balanced NASA sky, clean spacetime grid, cinematic accretion, and reference-style side panels.
# Dependencies: pip install pygame moderngl numpy pillow
# Run: python black_hole_kerr_reference_simulator_science_calibrated_cinematic.py
#
# Scientific note: this is an optical/cinematic gravitational-lensing visualization,
# not a direct EHT radio reconstruction and not a full GRMHD solver. The final
# picture intentionally keeps the polished Milky Way/stars/disk palette, while
# hidden physical fields now guide the render: Kerr ISCO/horizon scaling,
# capped transverse null-ray bending, g^3 redshift/Doppler cues, optical-depth
# hints, magnetic/synchrotron modulation, and critical-impact photon-ring weights.

import sys
import math
from pathlib import Path
from dataclasses import dataclass

import moderngl
import numpy as np
import pygame
from PIL import Image


WIDTH, HEIGHT = 1536, 864
FPS = 60

# Preferred NASA all-sky asset. Put the downloaded file beside this script.
# NOTE: .opdownload is usually an incomplete Opera/Chromium download. The script
# will try to read it, but a completed/renamed .exr, .png, or .jpg is strongly preferred.
SKY_TEXTURE_CANDIDATES = (
    "starmap_2020_16k_gal.exr",
    "starmap_2020_16k_gal.exr.opdownload",
    "starmap_2020_16k_gal.png",
    "starmap_2020_16k_gal.jpg",
    "milky_way_4k.jpg",
    "milky_way_sky_generated.png",
)

# 16K EXR is excellent, but uploading full 16K to the GPU can consume hundreds of MB.
# This keeps the web-demo/runtime fast while preserving much more detail than 4K.
SKY_MAX_UPLOAD_SIZE = (4096, 2048)
SKY_MIN_RUNTIME_SIZE = (4096, 2048)
FALLBACK_SKY_PATH = Path(__file__).with_name("milky_way_sky_generated.png")


VERTEX_SHADER = """
#version 330

in vec2 in_pos;
out vec2 v_uv;

void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 330

uniform vec2  u_resolution;
uniform float u_time;

uniform float u_mass;
uniform float u_spin;
uniform float u_isco;
uniform float u_horizon_kerr;
uniform float u_accretion;
uniform float u_yaw;
uniform float u_pitch;
uniform float u_distance;

uniform float u_show_disk;
uniform float u_show_ring;
uniform float u_show_grid;
uniform float u_show_stars;
uniform float u_show_particles;
uniform sampler2D u_background;
uniform float u_exposure;
uniform float u_contrast;
uniform float u_saturation;
uniform float u_bloom;
uniform float u_science_overlay;

in vec2 v_uv;
out vec4 fragColor;

#define PI 3.14159265359

float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;

    for (int i = 0; i < 6; i++) {
        v += a * noise(p);
        p *= 2.02;
        a *= 0.5;
    }

    return v;
}

vec3 hotColor(float x) {
    x = clamp(x, 0.0, 1.0);

    vec3 black = vec3(0.025, 0.010, 0.003);
    vec3 red   = vec3(0.70, 0.20, 0.035);
    vec3 gold  = vec3(1.00, 0.56, 0.16);
    vec3 white = vec3(1.00, 0.90, 0.68);

    vec3 c = mix(black, red, smoothstep(0.00, 0.34, x));
    c = mix(c, gold, smoothstep(0.30, 0.72, x));
    c = mix(c, white, smoothstep(0.66, 1.00, x));

    return c;
}

vec3 tonemap(vec3 c) {
    // Filmic-ish tonemap with less black crushing than the old curve.
    c = max(c, vec3(0.0));
    c = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);
    return pow(clamp(c, 0.0, 1.0), vec3(0.92));
}

vec3 sampleSkyTexture(vec2 uv) {
    // The NASA all-sky map already contains a real stellar distribution.
    // Sample a low mip level and compress highlights so the background reads
    // as deep space instead of a noisy wall of bright points.
    vec3 c0 = textureLod(u_background, uv, 1.15).rgb;
    vec3 c1 = textureLod(u_background, uv, 2.25).rgb;
    vec3 c = mix(c0, c1, 0.18);

    c = pow(clamp(c, 0.0, 1.0), vec3(2.2));
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));

    // Compress compact bright stars more than diffuse Milky Way dust.
    float highlightCompress = mix(1.0, 0.42, smoothstep(0.055, 0.46, luma));
    c *= highlightCompress;

    // Keep the galactic band visible, but less harsh and less crowded.
    c = mix(c, vec3(luma), 0.08);
    return c;
}

vec3 cameraPosition() {
    return vec3(
        sin(u_yaw) * cos(u_pitch),
        sin(u_pitch),
        cos(u_yaw) * cos(u_pitch)
    ) * u_distance;
}

mat3 cameraBasis(vec3 ro) {
    vec3 forward = normalize(-ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = normalize(cross(forward, right));
    return mat3(right, up, forward);
}

vec3 getRay(vec2 p, vec3 ro) {
    mat3 cam = cameraBasis(ro);
    return normalize(cam * vec3(p, 1.88));
}

vec3 starfield(vec3 rd) {
    if (u_show_stars < 0.5) return vec3(0.0);

    vec3 d = normalize(rd);

    float yaw = -0.34 + 0.008 * sin(u_time * 0.014);
    float roll = 0.58;
    mat3 ry = mat3(
        cos(yaw), 0.0, sin(yaw),
        0.0,      1.0, 0.0,
       -sin(yaw), 0.0, cos(yaw)
    );
    mat3 rz = mat3(
        cos(roll), -sin(roll), 0.0,
        sin(roll),  cos(roll), 0.0,
        0.0,        0.0,       1.0
    );
    vec3 skyDir = normalize(rz * ry * d);

    vec2 uv = vec2(
        atan(skyDir.z, skyDir.x) / (2.0 * PI) + 0.5,
        asin(skyDir.y) / PI + 0.5
    );

    // Photographic base sky: intentionally restrained so it supports the black hole.
    vec3 col = sampleSkyTexture(uv) * 0.78;

    // Subtle dust structure, not extra star snow.
    vec3 g = normalize(vec3(
        d.x * 0.82 + d.y * 0.35 - d.z * 0.45,
       -d.x * 0.30 + d.y * 0.92 + d.z * 0.24,
        d.x * 0.48 - d.y * 0.16 + d.z * 0.86));
    float plane = exp(-pow(abs(g.y) * 3.85, 2.0));
    float planeThin = exp(-pow(abs(g.y) * 10.5, 2.0));

    float cloudA = fbm(uv * vec2(4.8, 2.1) + vec2(0.17, 0.31));
    float cloudB = fbm(uv * vec2(12.5, 5.8) + vec2(-0.41, 0.09));
    float lane = smoothstep(0.53, 0.92, fbm(uv * vec2(19.0, 8.0) + vec2(1.4, 2.1)));

    col += vec3(0.014, 0.019, 0.034) * cloudA * plane * 0.13;
    col += vec3(0.042, 0.027, 0.016) * cloudB * planeThin * 0.08;
    col *= 1.0 - lane * plane * 0.18;

    // Sparse procedural enhancement. These are only rare highlights layered over
    // the NASA map, avoiding the fake crowded/snowy look.
    for (int layer = 0; layer < 2; layer++) {
        float scale = layer == 0 ? 460.0 : 920.0;
        float threshold = layer == 0 ? 0.99905 : 0.99955;

        vec2 p = uv * vec2(scale, scale * 0.54);
        vec2 id = floor(p);
        vec2 f = fract(p);
        float rnd = hash21(id + float(layer) * 27.31);
        float star = smoothstep(threshold, 1.0, rnd);

        vec2 offset = vec2(hash21(id + 3.1), hash21(id + 9.7)) - 0.5;
        float dist = length(f - 0.5 - offset * 0.34);
        float core = exp(-dist * (layer == 0 ? 32.0 : 52.0));
        float halo = exp(-dist * 8.5) * 0.040;

        float temp = hash21(id + 8.3);
        vec3 starCol = mix(vec3(0.62, 0.74, 1.0), vec3(1.0, 0.94, 0.82), smoothstep(0.08, 0.64, temp));
        starCol = mix(starCol, vec3(1.0, 0.68, 0.35), smoothstep(0.78, 1.0, temp));

        float boost = 0.70 + planeThin * 0.30;
        float brightness = mix(0.035, 0.45, hash21(id + 4.7)) * boost;
        col += star * starCol * brightness * (core + halo);
    }

    // Rare bright foreground stars: soft, not cross-shaped or over-bloomed.
    vec2 bp = uv * vec2(185.0, 98.0);
    vec2 bid = floor(bp);
    vec2 bf = fract(bp);
    float br = hash21(bid + 91.7);
    float brightStar = smoothstep(0.99965, 1.0, br);
    vec2 boff = vec2(hash21(bid + 12.4), hash21(bid + 29.1)) - 0.5;
    vec2 bq = bf - 0.5 - boff * 0.30;
    float bdist = length(bq);
    vec3 brightCol = mix(vec3(0.64, 0.76, 1.0), vec3(1.0, 0.70, 0.36), hash21(bid + 11.0));
    col += brightStar * brightCol * exp(-bdist * 32.0) * 0.34;
    col += brightStar * brightCol * exp(-bdist * 9.0) * 0.075;

    return col;
}

float gridLine(float x, float w) {
    float g = abs(fract(x) - 0.5);
    return 1.0 - smoothstep(w * 0.35, w, g);
}

vec3 sampleGrid(vec3 p) {
    if (u_show_grid < 0.5) return vec3(0.0);

    float horizon = 0.62;
    float r = length(p.xz) + 0.0001;
    if (r < horizon * 1.08) return vec3(0.0);

    // Clean embedding-diagram sheet inspired by the second reference: visible,
    // organized, and low-emission rather than neon/cyan beams.
    float well = -0.82 * (horizon * 2.10) / (r + horizon * 1.20);
    float d = abs(p.y - well);

    float sheetCore = exp(-(d * d) / (2.0 * 0.026 * 0.026));
    float sheetSoft = exp(-d * 15.0) * 0.11;
    float sheet = sheetCore + sheetSoft;

    float pull = 0.38 * horizon * horizon / (r * r + horizon * horizon);
    vec2 q = p.xz * (1.0 - pull);

    float spacing = 0.32;
    float gx = gridLine(q.x / spacing, 0.055);
    float gz = gridLine(q.y / spacing, 0.055);
    float majorX = gridLine(q.x / (spacing * 2.0), 0.035);
    float majorZ = gridLine(q.y / (spacing * 2.0), 0.035);

    float minor = max(gx, gz) * 0.58;
    float major = max(majorX, majorZ) * 0.80;
    float line = max(minor, major);

    float fadeOuter = 1.0 - smoothstep(1.2, 7.2, r);
    float fadeInner = smoothstep(horizon * 1.10, horizon * 2.10, r);
    float wellGlow = exp(-pow(r - horizon * 1.75, 2.0) / 0.50) * 0.055;

    vec3 minorCol = vec3(0.040, 0.078, 0.105);
    vec3 majorCol = vec3(0.065, 0.135, 0.175);
    vec3 col = mix(minorCol, majorCol, smoothstep(0.45, 0.88, major));

    return (col * line * sheet * 1.22 + vec3(0.030, 0.080, 0.105) * wellGlow * sheet)
           * fadeOuter * fadeInner;
}

float visualMassScale() {
    // Keep the mass slider meaningful for the UI/readout while preventing the
    // apparent angular size from exploding. This is equivalent to keeping the
    // camera framing calibrated as mass changes.
    return clamp(u_mass / 9.87, 0.80, 1.15);
}

float sceneDiskInnerRadius(float visualHorizon) {
    // Kerr prograde ISCO in GM/c^2 is supplied by the CPU. The visual mapping is
    // intentionally subtle so spin informs the inner edge without destroying the
    // existing cinematic silhouette.
    float t = clamp((u_isco - 1.237) / (6.0 - 1.237), 0.0, 1.0);
    return visualHorizon * mix(1.14, 1.24, t);
}

float physicalDiskRadius(float rScene, float innerScene) {
    // Scene radius -> physical radius in GM/c^2, anchored at the spin-dependent
    // ISCO. This hidden coordinate drives orbital velocity, temperature, and g.
    return max(u_horizon_kerr + 0.05, u_isco * rScene / max(innerScene, 0.001));
}

float keplerianOmega(float rPhys) {
    // Prograde Kerr circular-orbit angular frequency in geometrized units.
    return 1.0 / (pow(max(rPhys, 1.001), 1.5) + clamp(u_spin, 0.0, 0.998));
}

float thinDiskTemperature(float rScene, float innerScene) {
    // Shakura-Sunyaev / Novikov-Thorne inspired radial profile, normalized for
    // color and modulation rather than absolute Kelvin.
    float x = max(rScene / max(innerScene, 0.001), 1.001);
    float noTorque = pow(max(1.0 - inversesqrt(x), 0.0), 0.25);
    return clamp(pow(x, -0.75) * noTorque * 2.55, 0.0, 1.20);
}

float redshiftFactor(vec3 orbitalDir, vec3 rd, float rPhys) {
    // Controlled approximation of g = (k_mu u_obs^mu)/(k_mu u_em^mu).
    // It combines orbital Doppler shift with a Schwarzschild-like redshift term.
    float a = clamp(u_spin, 0.0, 0.998);
    float beta = clamp(sqrt(1.0 / max(rPhys, 2.20)) * (0.56 + 0.10 * a), 0.025, 0.62);
    float gamma = inversesqrt(max(0.001, 1.0 - beta * beta));
    float mu = dot(orbitalDir, -rd);
    float doppler = 1.0 / max(0.12, gamma * (1.0 - beta * mu));
    float grav = sqrt(clamp(1.0 - 1.62 / max(rPhys, 1.82), 0.22, 1.0));
    return clamp(doppler * grav, 0.34, 2.20);
}

float relativisticBeaming(vec3 orbitalDir, vec3 rd, float rPhys) {
    // Frequency-specific synchrotron intensity follows I_nu ~ g^3. We blend the
    // physical factor into unity so the current cinematic exposure is preserved.
    float g = redshiftFactor(orbitalDir, rd, rPhys);
    return clamp(mix(1.0, pow(g, 3.0), 0.58), 0.42, 3.55);
}

float criticalImpactParam() {
    // Scene-space proxy for the Schwarzschild/Kerr critical curve. It is mapped
    // to the current preferred photon-ring radius rather than raw screen pixels.
    float horizon = 0.62;
    return horizon * (1.18 - 0.022 * clamp(u_spin, 0.0, 0.998));
}

float geodesicWindingFromImpact(float b) {
    // Near the critical impact parameter, null rays wind around the hole. This
    // logarithmic proxy lets higher-order lensing cues become thinner and dimmer.
    float bc = criticalImpactParam();
    float eps = abs(b - bc) / max(bc, 0.0001);
    return clamp(-log(max(eps, 0.0016)) * 0.46, 0.0, 3.8);
}

float photonCriticalWeight(float b, float width) {
    float bc = criticalImpactParam();
    float critical = exp(-pow((b - bc) / max(width, 0.0001), 2.0));
    float winding = geodesicWindingFromImpact(b);
    return critical * exp(-0.62 * winding);
}

float synchrotronTransferWeight(float ne, float thetaE, float bMag, float pitchAngle) {
    // Lightweight optically thin synchrotron emissivity. It modulates the
    // cinematic disk instead of replacing it: j_nu ~ ne B^(3/2) exp[-sqrt(nu/nu_c)].
    float nu = 1.0;
    float nuC = 0.050 + 0.52 * bMag * thetaE * thetaE * pitchAngle;
    float jnu = ne * pow(max(bMag * pitchAngle, 0.001), 1.5) * exp(-sqrt(nu / max(nuC, 0.0001)));
    return clamp(0.75 + 0.45 * jnu, 0.70, 1.35);
}

float opticalDepthCue(float ne, float thetaE, float bMag) {
    // A small absorption/optical-depth cue. It prevents every dense lane from
    // becoming pure glow while keeping the render bright and readable.
    float alpha = ne * sqrt(max(bMag, 0.001)) / (thetaE * thetaE + 0.30);
    return clamp(alpha, 0.0, 1.0);
}

vec3 advanceNullGeodesic(vec3 pos, vec3 dir, float ds) {
    // Fast weak-field null-ray step with transverse deflection. It is not a full
    // Kerr geodesic integrator, but it preserves the geometric rule that gravity
    // bends the photon direction perpendicular to its current propagation.
    float horizon = 0.62;
    float r = length(pos) + 0.0001;
    float b = length(cross(pos, dir)) + 0.0001;
    float massScale = visualMassScale();

    vec3 towardBH = -pos / r;
    vec3 transverse = towardBH - dir * dot(towardBH, dir);
    float tLen = length(transverse);
    if (tLen > 0.0001) {
        transverse /= tLen;
    }

    float photonShell = exp(-pow(r - horizon * 1.50, 2.0) / 0.078);
    float nearShell = exp(-pow(r - horizon * 1.11, 2.0) / 0.030);
    float critical = photonCriticalWeight(b, horizon * 0.22);

    float bend = 0.040 * massScale / (r * r + 0.18);
    bend += 0.044 * photonShell + 0.018 * nearShell + 0.010 * critical / (r + 0.44);

    vec3 spinAxis = vec3(0.0, 1.0, 0.0);
    vec3 drag = cross(spinAxis, pos);
    drag -= dir * dot(drag, dir);
    float dLen = length(drag);
    vec3 frameDrag = dLen > 0.0001 ? drag / dLen : vec3(0.0);
    float dragAmount = 0.0052 * clamp(u_spin, 0.0, 0.998) * massScale / (r * r + 0.42);

    return normalize(dir + transverse * bend * ds + frameDrag * dragAmount * ds);
}

vec3 sampleDisk(vec3 p, vec3 rd) {
    if (u_show_disk < 0.5) return vec3(0.0);

    float horizon = 0.62;
    float inner = sceneDiskInnerRadius(horizon);
    float outer = 3.35;

    if (length(p) < horizon * 1.03) return vec3(0.0);

    float tilt = 0.18;
    vec3 diskNormal = normalize(vec3(0.0, cos(tilt), sin(tilt)));
    vec3 diskX = vec3(1.0, 0.0, 0.0);
    vec3 diskZ = normalize(cross(diskX, diskNormal));

    float x = dot(p, diskX);
    float y = dot(p, diskNormal);
    float z = dot(p, diskZ);

    float r = length(vec2(x, z));
    float a = atan(z, x);

    if (r < inner || r > outer) return vec3(0.0);

    // Geometrically thin emitting disk. The old broad vertical corona could stack
    // into a polar-looking light column. Here the disk remains thin, while a
    // faint lifted ribbon supplies the curved secondary image expected from
    // gravitational lensing of the disk behind the black hole.
    float qDisk = max(r - inner, 0.0);
    float thickness = 0.064 + 0.034 * exp(-1.22 * qDisk) + 0.010 * smoothstep(1.25, 2.8, r);
    float vertical = exp(-(y * y) / (2.0 * thickness * thickness));
    float lensLift = 0.115 + 0.145 * exp(-pow(r - horizon * 1.52, 2.0) / 0.22);
    float lensedSkin = exp(-pow(y - lensLift, 2.0) / (2.0 * (thickness * 0.72) * (thickness * 0.72)))
                     * exp(-pow(r - horizon * 1.58, 2.0) / 0.30) * 0.165;
    float corona = exp(-abs(y) / (thickness * 1.70)) * exp(-0.95 * qDisk) * 0.050;
    vertical = max(vertical, lensedSkin + corona);

    // Dark central cavity: avoids the unrealistic dusty fog around the hole.
    float cavity = smoothstep(inner, inner + 0.23, r);
    float radial = exp(-1.05 * max(r - inner, 0.0)) * cavity;

    float rPhys = physicalDiskRadius(r, inner);
    float omega = keplerianOmega(rPhys);
    float tempPhys = thinDiskTemperature(r, inner);

    float swirl = a + 0.95 / (r + 0.12) + u_time * (1.85 * omega + 0.10 + 0.18 * u_spin);
    vec2 flowUV = vec2(r * 6.8 - u_time * (0.16 + 0.38 * omega), swirl * 4.25);
    float turbulent = fbm(flowUV);
    float fineDust = fbm(flowUV * 3.7 + vec2(4.1, -1.7));
    float microDust = fbm(flowUV * 9.8 + vec2(-2.3, 5.2));
    float shear = sin(92.0 * r - 16.0 * a - u_time * (3.7 + 1.5 * u_spin));
    float filamentNoise = smoothstep(0.46, 0.92, microDust) * (0.42 + 0.58 * abs(shear));

    // Narrow luminous ring structure: photon-orbit rim + hot inner accretion rings.
    float ring1 = exp(-pow(r - 0.88, 2.0) / 0.0038);
    float ring2 = exp(-pow(r - 1.08, 2.0) / 0.0070);
    float ring3 = exp(-pow(r - 1.42, 2.0) / 0.0180);
    float ringTexture = 0.72 + 0.20 * sin(96.0 * r + 11.0 * a - u_time * 2.4) + 0.16 * turbulent + 0.12 * filamentNoise;
    float rings = (ring1 * 1.30 + ring2 * 0.62 + ring3 * 0.28) * ringTexture;

    float armA = sin(22.0 * r - 5.8 * a - u_time * 1.85 + turbulent * 3.4);
    float armB = sin(44.0 * r + 3.9 * a - u_time * 1.20 + fineDust * 2.2);
    float armC = sin(82.0 * r - 9.0 * a - u_time * 2.60 + microDust * 1.4);
    float streaks = smoothstep(0.20, 1.0, abs(armC)) * filamentNoise;
    float spiral = 0.50 + 0.20 * turbulent + 0.12 * armA + 0.075 * armB + 0.050 * streaks + rings;

    // Absorbing lanes inside the accretion flow, not a visible fog around the hole.
    float lane = smoothstep(0.58, 0.94, fineDust) * (0.35 + 0.65 * smoothstep(1.05, 2.2, r));
    spiral *= 1.0 - lane * 0.22;
    spiral += filamentNoise * (1.0 - smoothstep(0.82, 2.9, r)) * 0.18;

    // Tiny star-like hot knots embedded inside bright rings and accretion lanes.
    vec2 cell = vec2(a * 34.0 + r * 8.0 - u_time * 0.70, r * 24.0);
    vec2 cid = floor(cell);
    vec2 cf = fract(cell);
    float cr = hash21(cid + 44.2);
    float clumpMask = smoothstep(0.946, 1.0, cr);
    vec2 coff = vec2(hash21(cid + 2.0), hash21(cid + 7.0));
    float cdist = length(cf - coff);
    float ringPreference = ring1 * 0.90 + ring2 * 0.70 + (1.0 - smoothstep(0.95, 2.6, r)) * 0.35;
    float clumps = clumpMask * exp(-cdist * 24.0) * ringPreference;

    vec3 orbitalDir = normalize(-sin(a) * diskX + cos(a) * diskZ);
    vec3 radialDir = normalize(cos(a) * diskX + sin(a) * diskZ);
    float g = redshiftFactor(orbitalDir, rd, rPhys);
    float doppler = relativisticBeaming(orbitalDir, rd, rPhys);

    // Hidden magnetic/synchrotron layer. It modulates the existing cinematic
    // brightness, preserving the gold/white look while tying texture to plasma
    // density, magnetic pitch angle, and redshift.
    float toroidalStrength = 0.88 + 0.10 * clamp(u_spin, 0.0, 0.998);
    float poloidalStrength = 0.13 + 0.09 * smoothstep(0.20, 1.80, qDisk) + 0.025 * sin(2.5 * a + u_time * 0.18);
    vec3 bField = normalize(orbitalDir * toroidalStrength + diskNormal * poloidalStrength + radialDir * (0.045 * (turbulent - 0.5)));
    float pitchAngle = clamp(length(cross(bField, -rd)), 0.065, 1.0);
    float ne = vertical * radial * (0.54 + 0.46 * turbulent + 0.28 * ring1 + 0.16 * ring2);
    float thetaE = clamp(0.32 + 1.50 * pow(tempPhys, 1.70) + 0.18 * ring1 + 0.08 * filamentNoise, 0.10, 2.60);
    float bMag = pow(max(rPhys, 1.15), -1.04) * (1.0 + 0.36 * ring1 + 0.14 * ring2 + 0.08 * filamentNoise);
    float synchMod = synchrotronTransferWeight(ne, thetaE, bMag, pitchAngle);
    float tau = opticalDepthCue(ne, thetaE, bMag);
    float redshiftMod = clamp(pow(g, 3.0), 0.55, 1.85);

    float heat = mix(exp(-0.68 * max(r - inner, 0.0)), tempPhys, 0.18);
    float photonBoost = 1.0 + 0.55 * ring1 + 0.28 * ring2;
    float physicalMod = synchMod * mix(1.0, redshiftMod, 0.28) * (1.0 - tau * lane * 0.075);

    float brightness = vertical * radial * spiral * doppler * heat * photonBoost * physicalMod * u_accretion * 1.22;
    brightness += vertical * clumps * doppler * heat * physicalMod * 1.48 * u_accretion;
    brightness += vertical * filamentNoise * radial * doppler * heat * 0.25 * physicalMod * u_accretion;
    brightness = pow(max(brightness, 0.0), 0.56);

    vec3 base = hotColor(clamp(brightness + tempPhys * 0.10 + max(g - 1.0, 0.0) * 0.035, 0.0, 1.0)) * brightness;

    vec3 whiteRim = vec3(1.0, 0.94, 0.76) * vertical * ring1 * doppler * physicalMod * 0.74 * u_accretion;
    vec3 ringSpark = vec3(1.0, 0.84, 0.55) * clumps * vertical * doppler * physicalMod * 0.64;
    vec3 coldLane = vec3(0.22, 0.095, 0.040) * vertical * radial * lane * (0.09 + tau * 0.035);
    vec3 blueShock = vec3(0.55, 0.68, 1.0) * clumps * max(g - 1.16, 0.0) * pitchAngle * 0.18;

    return base + whiteRim + ringSpark + blueShock + coldLane;
}

vec3 samplePhotonHalo(vec3 p, vec3 rd) {
    if (u_show_ring < 0.5) return vec3(0.0);

    float horizon = 0.62;
    float critical = criticalImpactParam();
    float inner = 0.70;
    float outer = 2.05;

    float tilt = 0.18;
    vec3 diskNormal = normalize(vec3(0.0, cos(tilt), sin(tilt)));
    vec3 diskX = vec3(1.0, 0.0, 0.0);
    vec3 diskZ = normalize(cross(diskX, diskNormal));

    float x = dot(p, diskX);
    float y = dot(p, diskNormal);
    float z = dot(p, diskZ);
    float r = length(vec2(x, z)) + 0.0001;
    float a = atan(z, x);
    float sphericalR = length(p);

    if (sphericalR < horizon * 1.02 || r < inner || r > outer) return vec3(0.0);

    // Curved lensed photon ribbon. The previous broad upper sheath behaved like
    // a vertical polar glow. A physical accretion disk without a jet should instead
    // show a connected secondary image that bends around the photon region.
    float equatorialRibbon = exp(-(y * y) / (2.0 * 0.076 * 0.076));
    float arcLift = 0.070 + 0.245 * exp(-pow(r - critical * 1.305, 2.0) / 0.25);
    float arcWidth = 0.045 + 0.018 * smoothstep(horizon * 1.12, horizon * 2.00, r);
    float upperRibbon = exp(-pow(y - arcLift, 2.0) / (2.0 * arcWidth * arcWidth))
                      * exp(-pow(r - critical * 1.34, 2.0) / 0.32) * 0.72;
    float lowerRibbon = exp(-pow(y + arcLift * 0.52, 2.0) / (2.0 * (arcWidth * 0.88) * (arcWidth * 0.88)))
                      * exp(-pow(r - critical * 1.23, 2.0) / 0.28) * 0.26;
    float sideBridge = exp(-(y * y) / (2.0 * 0.115 * 0.115))
                     * exp(-pow(r - critical * 1.20, 2.0) / 0.115) * 0.38;
    float azimuthContinuity = 0.88 + 0.12 * cos(2.0 * a);
    float vertical = (equatorialRibbon * 0.88 + sideBridge + upperRibbon + lowerRibbon) * azimuthContinuity;

    float winding = geodesicWindingFromImpact(r);
    float photon = exp(-pow(r - critical, 2.0) / 0.0036) * (0.94 + 0.06 * winding);
    float innerFire = exp(-pow(r - critical * 1.17, 2.0) / 0.0105);
    float midBand = exp(-pow(r - critical * 1.39, 2.0) / 0.0260);
    float outerBand = exp(-pow(r - critical * 1.76, 2.0) / 0.0740) * exp(-0.35 * winding);

    float rPhys = max(u_horizon_kerr + 0.05, 2.0 + (r / horizon - 1.0) * 2.4);
    float omega = keplerianOmega(rPhys);
    float swirl = a + 1.15 / (r + 0.11) + u_time * (3.2 * omega + 0.24 + 0.22 * u_spin);
    vec2 uv = vec2(r * 12.0 - u_time * 0.42, swirl * 7.8);
    float n1 = fbm(uv);
    float n2 = fbm(uv * 2.7 + vec2(3.7, -2.2));
    float n3 = fbm(uv * 7.8 + vec2(-1.3, 4.1));

    float strandA = 0.5 + 0.5 * sin(122.0 * r + 19.0 * a - u_time * 3.8 + n1 * 4.0);
    float strandB = 0.5 + 0.5 * sin(207.0 * r - 31.0 * a - u_time * 5.4 + n2 * 2.8);
    float filaments = smoothstep(0.28, 0.96, strandA) * (0.55 + 0.45 * n1)
                    + smoothstep(0.62, 0.99, strandB) * (0.35 + 0.65 * n2);
    filaments *= 0.58 + 0.42 * smoothstep(0.42, 0.92, n3);

    vec3 orbitalDir = normalize(-sin(a) * diskX + cos(a) * diskZ);
    float g = redshiftFactor(orbitalDir, rd, rPhys);
    float doppler = relativisticBeaming(orbitalDir, rd, rPhys);
    float criticalWeight = photonCriticalWeight(r, critical * 0.10);

    float shell = (photon * 1.30 + innerFire * 0.78 + midBand * 0.46 + outerBand * 0.14);
    shell *= 0.94 + 0.14 * criticalWeight;
    float brightness = vertical * shell * (0.54 + 0.58 * filaments) * doppler * mix(1.0, clamp(pow(g, 3.0), 0.60, 1.75), 0.22) * u_accretion;

    // Plasma beads/sparks embedded inside the 3D ring itself.
    vec2 cell = vec2(a * 78.0 + r * 13.0 - u_time * (1.05 + 0.45 * u_spin), r * 44.0 + y * 8.0);
    vec2 id = floor(cell);
    vec2 f = fract(cell);
    vec2 off = vec2(hash21(id + 5.1), hash21(id + 17.3));
    float rnd = hash21(id + 41.7);
    float bead = smoothstep(0.972, 1.0, rnd) * exp(-length(f - off) * 34.0);
    bead *= (photon * 1.0 + innerFire * 0.85 + midBand * 0.62) * vertical * u_show_particles;

    vec3 col = vec3(1.0, 0.56, 0.18) * brightness * 0.92;
    col += vec3(1.0, 0.90, 0.62) * photon * vertical * doppler * (0.31 + 0.15 * filaments) * u_accretion;
    col += vec3(1.0, 0.78, 0.38) * bead * doppler * 1.38 * u_accretion;
    col += vec3(1.0, 0.31, 0.08) * outerBand * vertical * filaments * 0.095 * doppler * u_accretion;

    // Clean central cavity so the shadow remains black and the ring reads as
    // surrounding the horizon instead of filling it.
    col *= smoothstep(horizon * 1.04, horizon * 1.34, sphericalR);
    return col;
}

vec3 sampleLensRim(vec3 ro, vec3 rd, vec2 p, float closestImpact) {
    if (u_show_ring < 0.5) return vec3(0.0);

    float horizon = 0.62;
    float critical = criticalImpactParam();
    float winding = geodesicWindingFromImpact(closestImpact);
    vec3 closest = ro + rd * dot(-ro, rd);

    float tilt = 0.18;
    vec3 diskNormal = normalize(vec3(0.0, cos(tilt), sin(tilt)));
    vec3 diskX = vec3(1.0, 0.0, 0.0);
    vec3 diskZ = normalize(cross(diskX, diskNormal));

    float x = dot(closest, diskX);
    float y = dot(closest, diskNormal);
    float z = dot(closest, diskZ);
    float a = atan(z, x);
    float diskSide = dot(normalize(cross(rd, diskX) + vec3(0.0001)), diskNormal);
    float theta = atan(y * 1.15, x);
    float ringCoord = theta / (2.0 * PI) + 0.5;

    float noiseArc = fbm(vec2(ringCoord * 28.0 + u_time * 0.11, closestImpact * 18.0));
    float broken = smoothstep(0.18, 0.98, 0.5 + 0.5 * sin(148.0 * closestImpact + 17.0 * a - u_time * 2.4 + noiseArc * 4.2));
    broken *= 0.55 + 0.45 * noiseArc;

    float rim = exp(-pow(closestImpact - critical * 0.915, 2.0) / 0.0028) * (0.94 + 0.06 * winding);
    float innerFire = exp(-pow(closestImpact - critical * 1.085, 2.0) / 0.0110);
    float outerArc = exp(-pow(closestImpact - critical * 1.373, 2.0) / 0.0500) * exp(-0.18 * winding);

    // Unlike the old p.y mask, this modulation is derived from the ray's
    // closest world-space pass around the hole and the disk frame.
    float diskFrameGate = smoothstep(0.02, 0.62, abs(x)) * (0.72 + 0.28 * smoothstep(-0.35, 0.35, y * diskSide));
    float asym = 1.0 + 0.36 * tanh(-x * 1.4 + u_spin * 0.45);

    vec3 col = vec3(0.0);
    col += vec3(1.0, 0.94, 0.76) * rim * (0.84 + 0.16 * broken) * 0.34 * u_accretion;
    col += vec3(1.0, 0.66, 0.25) * innerFire * (0.055 + 0.090 * broken) * diskFrameGate * asym * u_accretion;
    col += vec3(1.0, 0.40, 0.11) * outerArc * broken * diskFrameGate * asym * 0.058 * u_accretion;

    // A controlled upper lensed arc: connected to the same impact-parameter ring
    // and lifted only slightly above the disk plane, so it curves around the hole
    // instead of rising as two independent vertical curtains.
    float liftedCenter = 0.090 + 0.080 * exp(-pow(closestImpact - critical * 1.203, 2.0) / 0.038);
    float upperArcTrack = exp(-pow(y - liftedCenter, 2.0) / (2.0 * 0.070 * 0.070));
    float connectedArc = upperArcTrack * exp(-pow(closestImpact - critical * 1.220, 2.0) / 0.030)
                       * smoothstep(-0.02, 0.18, y) * (0.68 + 0.32 * broken) * diskFrameGate;
    col += vec3(1.0, 0.78, 0.38) * connectedArc * asym * 0.070 * u_accretion;

    if (u_show_particles > 0.5) {
        vec2 cell = vec2(ringCoord * 92.0 + u_time * (0.75 + u_spin), closestImpact * 36.0 + a * 1.7);
        vec2 id = floor(cell);
        vec2 f = fract(cell);
        vec2 off = vec2(hash21(id + 8.0), hash21(id + 19.0));
        float rnd = hash21(id + 61.0);
        float spark = smoothstep(0.984, 1.0, rnd) * exp(-length(f - off) * 38.0);
        col += vec3(1.0, 0.84, 0.50) * spark * (rim * 0.50 + innerFire * 0.80 + outerArc * 0.62) * diskFrameGate * 1.10 * u_accretion;
    }

    return col;
}


vec3 scienceCalibrationOverlay(vec3 col, vec2 p, float impactScreen, float impactPath) {
    if (u_science_overlay < 0.5) return col;

    float horizon = 0.62;
    float critical = criticalImpactParam();
    float inner = sceneDiskInnerRadius(horizon);
    float iscoMark = exp(-pow(impactScreen - inner, 2.0) / 0.0035);
    float horizonMark = exp(-pow(impactScreen - horizon, 2.0) / 0.0025);
    float photonMark = exp(-pow(impactScreen - critical, 2.0) / 0.0028);
    float pathMark = exp(-pow(impactPath - critical, 2.0) / 0.010);

    vec3 overlay = vec3(0.0);
    overlay += vec3(0.18, 0.62, 1.0) * iscoMark * 0.52;       // ISCO guide
    overlay += vec3(1.0, 0.24, 0.12) * horizonMark * 0.42;    // horizon scale
    overlay += vec3(0.95, 0.84, 0.34) * photonMark * 0.68;    // photon critical curve
    overlay += vec3(0.28, 1.0, 0.66) * pathMark * 0.18;       // bent-path closest approach

    float stripe = smoothstep(0.48, 0.54, fract((p.y + 1.0) * 18.0));
    overlay *= 0.72 + 0.28 * stripe;
    return mix(col, col + overlay, 0.86);
}


void main() {
    vec2 uv = v_uv;
    vec2 p = uv * 2.0 - 1.0;
    p.x *= u_resolution.x / u_resolution.y;

    vec3 ro = cameraPosition();
    vec3 rd = getRay(p, ro);

    float horizon = 0.62;

    vec3 rayPos = ro;
    vec3 rayDir = rd;

    vec3 emission = vec3(0.0);
    float opacity = 0.0;
    float pathMinImpact = 1e9;

    bool absorbed = false;

    float stepSize = 0.038;

    for (int i = 0; i < 232; i++) {
        float r = length(rayPos);
        float pathImpact = length(cross(rayPos, rayDir));
        pathMinImpact = min(pathMinImpact, pathImpact);

        if (r < horizon) {
            absorbed = true;
            break;
        }

        vec3 diskCol = sampleDisk(rayPos, rayDir);
        vec3 ringCol = samplePhotonHalo(rayPos, rayDir);
        vec3 gridCol = sampleGrid(rayPos);

        float localPower = length(diskCol) + length(ringCol) * 0.72;

        emission += diskCol * 0.071 * (1.0 - opacity);
        emission += ringCol * 0.086 * (1.0 - opacity * 0.62);
        opacity += localPower * 0.0080 * (1.0 - opacity);

        // The grid is diagnostic: let it survive moderate opacity so it remains
        // visible around the well without washing over the brightest disk core.
        emission += gridCol * 0.066 * (1.0 - opacity * 0.30);

        rayDir = advanceNullGeodesic(rayPos, rayDir, stepSize);
        rayPos += rayDir * stepSize;

        if (length(rayPos) > 9.0 || opacity > 0.96) break;
    }

    vec3 col = emission;

    float closestImpact = length(cross(ro, rd));
    float physicalClosestImpact = min(closestImpact, pathMinImpact);
    float criticalImpact = criticalImpactParam();

    // Strong gravity clears diffuse haze near the shadow. Point stars remain visible
    // because the attenuation is not a hard black overlay; it mainly suppresses fog.
    float shadowHalo = exp(-pow(closestImpact - horizon * 1.28, 2.0) / 0.19);
    float deepShadow = exp(-pow(closestImpact - horizon * 0.82, 2.0) / 0.035);

    if (!absorbed) {
        vec3 bg = starfield(rayDir) * 1.15;
        // Stronger photon-sphere attenuation makes stars visually wrap into arcs
        // instead of reading as a flat backdrop pasted behind the hole.
        bg *= 1.0 - shadowHalo * 0.20;
        bg *= 1.0 - deepShadow * 0.64;
        col += bg * (1.0 - opacity);
    }

    // Thin lensing rim, tied to the ray's closest world-space pass around the hole.
    // The detailed glowing structure itself is emitted by samplePhotonHalo() inside
    // the ray march, so it remains spatially attached to the black hole.
    col += sampleLensRim(ro, rd, p, closestImpact);

    // Final gravitational darkness. Keep it local, otherwise the whole scene becomes muddy.
    float bowl = exp(-pow(closestImpact - horizon * 1.20, 2.0) / 0.050);
    float ringProtect = exp(-pow(closestImpact - criticalImpact, 2.0) / 0.0080);
    col *= 1.0 - bowl * 0.34 * (1.0 - ringProtect * 0.72);

    if (absorbed) {
        // No background is added after absorption; keep accumulated foreground
        // ring/disk emission so front-side plasma can cross the silhouette naturally.
        col *= 0.88;
    }

    col = scienceCalibrationOverlay(col, p, closestImpact, physicalClosestImpact);

    float vignette = smoothstep(1.68, 0.18, length(p));
    col *= mix(0.48, 1.0, vignette);

    // Cheap HDR-style local glow. True multipass bloom is better, but this single-pass
    // bloom shoulder gives a brighter white-hot core without tanking frame rate.
    float preLuma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    vec3 hotBloom = max(col - vec3(0.92), vec3(0.0));
    col += hotBloom * (0.115 + 0.145 * u_bloom);
    col += vec3(1.0, 0.58, 0.18) * smoothstep(0.8, 2.8, preLuma) * 0.026 * u_bloom;
    col *= u_exposure;
    col = tonemap(col);
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(luma), col, u_saturation);
    col = (col - 0.5) * u_contrast + 0.5;
    col = clamp(col, 0.0, 1.0);

    fragColor = vec4(col, 1.0);
}

"""




def create_generated_milky_way(path: Path, width: int = 4096, height: int = 2048) -> None:
    """Create a lightweight equirectangular sky texture used as a fallback asset.

    Replace this PNG with a real 4K/8K public-domain Milky Way panorama for an even
    more photographic web-demo look. The shader treats the texture as distant light
    and gravitationally bends it around the black hole.
    """
    if path.exists():
        return

    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:height, 0:width]
    u = xx / width
    v = yy / height

    angle = -0.23
    x = (u - 0.5) * 2.0
    y = (v - 0.5) * 2.0
    band_y = np.sin(angle) * x + np.cos(angle) * y + 0.08 * np.sin(2.0 * np.pi * (u * 1.15 + 0.07))
    band_x = np.cos(angle) * x - np.sin(angle) * y

    plane = np.exp(-(np.abs(band_y) * 3.5) ** 2)
    thin = np.exp(-(np.abs(band_y) * 12.0) ** 1.6)
    core = np.exp(-((band_x + 0.34) ** 2 * 3.2 + band_y ** 2 * 48.0))

    def upsample_noise(low_h: int, low_w: int) -> np.ndarray:
        small = (rng.random((low_h, low_w)) * 255).astype(np.uint8)
        im = Image.fromarray(small, mode="L").resize((width, height), Image.Resampling.BICUBIC)
        return np.asarray(im).astype(np.float32) / 255.0

    n1 = sum(upsample_noise(max(8, height // s), max(16, width // s)) * a for s, a in [(96, 1.0), (48, 0.55), (24, 0.30), (12, 0.16)]) / 2.01
    n2 = sum(upsample_noise(max(8, height // s), max(16, width // s)) * a for s, a in [(64, 1.0), (32, 0.55), (16, 0.30)]) / 1.85

    lanes = np.clip((n2 - 0.48) * 4.0, 0.0, 1.0) * plane
    dust = plane * (0.30 + 0.70 * n1)

    img = np.zeros((height, width, 3), dtype=np.float32)
    img += np.array([0.010, 0.013, 0.025], dtype=np.float32)
    img += dust[..., None] * np.array([0.045, 0.060, 0.120], dtype=np.float32)
    img += thin[..., None] * (0.40 + 0.60 * n1[..., None]) * np.array([0.30, 0.17, 0.075], dtype=np.float32)
    img += core[..., None] * np.array([1.10, 0.65, 0.28], dtype=np.float32)
    img *= (1.0 - 0.62 * lanes[..., None])

    star_count = 28000
    sx = rng.integers(0, width, star_count)
    sy = rng.integers(0, height, star_count)
    local_plane = plane[sy, sx]
    keep = rng.random(star_count) < (0.23 + 0.77 * local_plane)
    sx, sy, local_plane = sx[keep], sy[keep], local_plane[keep]
    temps = rng.random(len(sx))
    colors = np.stack([
        0.65 + 0.45 * temps,
        0.72 + 0.25 * (1.0 - np.abs(temps - 0.5) * 2.0),
        0.88 + 0.22 * (1.0 - temps),
    ], axis=1)
    bright = (0.16 + rng.random(len(sx)) ** 5 * 2.9) * (0.55 + 1.65 * local_plane)
    np.add.at(img, (sy, sx), colors * bright[:, None])

    big = bright > 1.1
    for px, py, col, b in zip(sx[big], sy[big], colors[big], bright[big]):
        if 1 <= px < width - 1 and 1 <= py < height - 1:
            img[py-1:py+2, px-1:px+2] += col * b * 0.08

    for _ in range(46):
        gx = int(rng.integers(80, width - 80))
        gy = int(rng.integers(80, height - 80))
        radx = int(rng.integers(10, 38))
        rady = int(rng.integers(2, 9))
        tint = np.array([1.0, 0.72, 0.42]) if rng.random() < 0.55 else np.array([0.55, 0.68, 1.0])
        xs = np.arange(gx - radx * 2, gx + radx * 2)
        ys = np.arange(gy - radx * 2, gy + radx * 2)
        X, Y = np.meshgrid(xs, ys)
        a = rng.random() * np.pi
        qx = np.cos(a) * (X - gx) - np.sin(a) * (Y - gy)
        qy = np.sin(a) * (X - gx) + np.cos(a) * (Y - gy)
        mask = np.exp(-(qx ** 2 / (2 * radx ** 2) + qy ** 2 / (2 * rady ** 2)))
        vx = (xs >= 0) & (xs < width)
        vy = (ys >= 0) & (ys < height)
        img[ys[vy][:, None], xs[vx][None, :]] += mask[vy][:, vx][..., None] * tint * 0.22

    img = 1.0 - np.exp(-img * 1.22)
    img = np.clip(img ** (1.0 / 2.2), 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(path)


def find_sky_texture() -> Path:
    """Pick the best available equirectangular sky map next to the script."""
    root = Path(__file__).resolve().parent
    for name in SKY_TEXTURE_CANDIDATES:
        candidate = root / name
        if candidate.exists() and candidate.stat().st_size > 1024:
            return candidate

    # Be forgiving about NASA/browser names.
    for pattern in ("starmap_2020_16k_gal*", "*16k*gal*", "*milky*way*", "*.exr", "*.png", "*.jpg"):
        matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        if matches:
            return matches[0]

    return FALLBACK_SKY_PATH


def _array_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """Convert HDR/float or integer image arrays into a display-ready RGB panorama."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]

    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.max(initial=0.0) > 255.0 or arr.dtype.kind == "f":
        # NASA EXR maps are HDR/linear. Robust percentile exposure keeps the galactic
        # core bright without crushing faint star detail.
        positive = arr[arr > 0.0]
        if positive.size:
            white = float(np.percentile(positive, 99.72))
        else:
            white = 1.0
        white = max(white, 1e-6)
        arr = np.clip(arr / white, 0.0, 24.0)
        arr = 1.0 - np.exp(-arr * 1.35)
        arr = np.clip(arr, 0.0, 1.0) ** (1.0 / 2.2)
        arr = arr * 255.0
    else:
        arr = np.clip(arr, 0.0, 255.0)

    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _load_exr_or_hdr_image(path: Path) -> Image.Image:
    """Load EXR/HDR with optional backends, then convert it to an RGB PIL image."""
    errors = []

    try:
        import imageio.v3 as iio
        return _array_to_pil_rgb(iio.imread(path))
    except Exception as exc:
        errors.append(f"imageio: {exc}")

    try:
        import os
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError("cv2.imread returned None")
        if arr.ndim == 3:
            arr = arr[..., ::-1]
        return _array_to_pil_rgb(arr)
    except Exception as exc:
        errors.append(f"opencv: {exc}")

    raise RuntimeError(
        "Could not read the EXR sky map. Install imageio/opencv-python, or convert the NASA EXR to PNG/JPG first. "
        "Example: magick starmap_2020_16k_gal.exr starmap_2020_16k_gal.png. "
        + " | ".join(errors)
    )


def load_quality_sky_surface(path: Path) -> pygame.Surface:
    """Load a NASA 16K/8K/4K equirectangular sky map with GPU-friendly preprocessing."""
    suffixes = "".join(path.suffixes).lower()

    if path.name.endswith(".opdownload"):
        print("Warning: using a .opdownload file. If loading fails, wait for the download to finish and rename it to .exr.")

    if ".exr" in suffixes or ".hdr" in suffixes:
        image = _load_exr_or_hdr_image(path)
    else:
        image = Image.open(path).convert("RGB")

    # Keep aspect ratio and avoid huge VRAM usage from full 16K uploads.
    w, h = image.size
    max_w, max_h = SKY_MAX_UPLOAD_SIZE
    min_w, min_h = SKY_MIN_RUNTIME_SIZE
    scale_down = min(max_w / w, max_h / h, 1.0)
    scale_up = max(min_w / w, min_h / h, 1.0) if (w < min_w or h < min_h) else 1.0
    scale = scale_down if scale_down < 1.0 else scale_up

    if scale != 1.0:
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    from PIL import ImageEnhance, ImageFilter
    image = ImageEnhance.Brightness(image).enhance(0.88)
    image = ImageEnhance.Contrast(image).enhance(0.94)
    image = ImageEnhance.Sharpness(image).enhance(0.86)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.12))

    print(f"Loaded sky texture: {path.name} -> {image.size[0]}x{image.size[1]} GPU upload")
    data = image.tobytes("raw", "RGB")
    return pygame.image.frombuffer(data, image.size, "RGB").copy()


@dataclass
class State:
    mass: float = 9.87
    spin: float = 0.85
    accretion: float = 0.82
    time_scale: float = 1.0

    yaw: float = 0.0
    pitch: float = 0.16
    distance: float = 5.8

    show_disk: bool = True
    show_ring: bool = True
    show_grid: bool = True
    show_stars: bool = True
    show_particles: bool = True
    cinematic: bool = False
    science_overlay: bool = False
    show_ui: bool = True

    exposure: float = 1.30
    bloom: float = 1.55
    contrast: float = 1.06
    saturation: float = 0.94

    dragging: bool = False
    dragging_slider: str | None = None

def clamp(x, a, b):
    return max(a, min(b, x))


def kerr_horizon_radius(spin: float) -> float:
    """Outer Kerr horizon radius r_+ in units of GM/c^2."""
    a = clamp(spin, 0.0, 0.998)
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a))


def kerr_isco_radius(spin: float) -> float:
    """Prograde Kerr ISCO radius in units of GM/c^2."""
    a = clamp(spin, 0.0, 0.998)
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    term = max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    return 3.0 + z2 - math.sqrt(term)



def draw_text(screen, font, txt, x, y, color=(235, 235, 235)):
    screen.blit(font.render(txt, True, color), (x, y))


UI_TITLE = (238, 238, 238)
UI_TEXT = (230, 230, 232)
UI_MUTED = (185, 188, 194)
UI_PANEL_BG = (16, 16, 18, 178)
UI_PANEL_BORDER = (105, 108, 116, 86)
UI_LINE = (255, 255, 255, 18)
UI_SLIDER_TRACK = (54, 55, 60)
UI_SLIDER_FILL = (145, 147, 152)
UI_SLIDER_KNOB = (215, 216, 220)

LEFT_X = 12
RIGHT_X = 1304
PANEL_W = 220
SLIDER_X = RIGHT_X + 16
SLIDER_W = 172
CHECK_X = RIGHT_X + 16


def draw_panel(screen, rect, title=None):
    surf = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(surf, UI_PANEL_BG, surf.get_rect(), border_radius=8)
    pygame.draw.rect(surf, UI_PANEL_BORDER, surf.get_rect(), 1, border_radius=8)
    pygame.draw.line(surf, UI_LINE, (1, 33), (rect[2] - 2, 33), 1)
    screen.blit(surf, rect[:2])
    if title:
        title_font = pygame.font.SysFont("arial", 14, bold=True)
        draw_text(screen, title_font, title, rect[0] + 14, rect[1] + 14, UI_TITLE)


def draw_checkbox(screen, font, label, checked, x, y):
    box = pygame.Rect(x, y, 16, 16)
    pygame.draw.rect(screen, (215, 216, 220) if checked else (18, 18, 20), box, border_radius=3)
    pygame.draw.rect(screen, (232, 232, 235), box, 1, border_radius=3)
    if checked:
        pygame.draw.line(screen, (20, 20, 22), (x + 3, y + 8), (x + 7, y + 12), 2)
        pygame.draw.line(screen, (20, 20, 22), (x + 7, y + 12), (x + 14, y + 3), 2)
    draw_text(screen, font, label, x + 26, y - 1, UI_TEXT)


def draw_slider(screen, font, label, value, x, y, minv, maxv, width=172):
    draw_text(screen, font, label, x, y, UI_TEXT)
    draw_text(screen, font, f"{value:.2f}", x + width - 44, y, UI_TEXT)
    bar_y = y + 30
    pygame.draw.rect(screen, UI_SLIDER_TRACK, (x, bar_y, width, 6), border_radius=4)
    f = clamp((value - minv) / (maxv - minv), 0.0, 1.0)
    knob_x = int(x + f * width)
    pygame.draw.rect(screen, UI_SLIDER_FILL, (x, bar_y, max(2, knob_x - x), 6), border_radius=4)
    pygame.draw.circle(screen, UI_SLIDER_KNOB, (knob_x, bar_y + 3), 7)


def draw_button(screen, font, label, rect):
    pygame.draw.rect(screen, (47, 47, 50, 225), rect, border_radius=6)
    pygame.draw.rect(screen, (118, 120, 126, 90), rect, 1, border_radius=6)
    tx = rect[0] + rect[2] // 2 - font.size(label)[0] // 2
    ty = rect[1] + rect[3] // 2 - font.get_height() // 2
    draw_text(screen, font, label, tx, ty, UI_TEXT)


def draw_status_table(screen, font, x, y, rows, col2=108, line=26):
    for i, (k, v) in enumerate(rows):
        yy = y + i * line
        draw_text(screen, font, k, x, yy, UI_TEXT)
        draw_text(screen, font, v, x + col2, yy, UI_TEXT)


def draw_ui(screen, font, small_font, state, fps, sim_time):
    if not state.show_ui:
        return

    # Left side: second-reference style compact translucent panels.
    draw_panel(screen, (LEFT_X, 52, 220, 178), "Simulation Info")
    frame_ms = 1000.0 / max(fps, 1.0)
    rows = [
        ("BH Mass:", f"{state.mass:.2f} M_sun"),
        ("Spin (a):", f"{state.spin:.2f}"),
        ("Accretion Rate:", f"{state.accretion:.2f}"),
        ("Time Scale:", f"{state.time_scale:.2f} x"),
        ("Frame Time:", f"{frame_ms:4.1f} ms ({fps:.0f} FPS)"),
    ]
    draw_status_table(screen, font, LEFT_X + 14, 96, rows, 104, 25)

    draw_panel(screen, (LEFT_X, 246, 220, 152), "Camera")
    rows = [
        ("Yaw:", f"{state.yaw * 180 / 3.14159:5.1f}°"),
        ("Pitch:", f"{state.pitch * 180 / 3.14159:5.1f}°"),
        ("Distance:", f"{state.distance:4.1f}"),
        ("FOV:", "28.2°"),
    ]
    draw_status_table(screen, font, LEFT_X + 14, 292, rows, 92, 25)

    draw_panel(screen, (LEFT_X, 412, 220, 360), "Controls")
    controls = [
        ("Mouse Drag", "Orbit"),
        ("Scroll", "Zoom"),
        ("A / D", "Orbit Left/Right"),
        ("Q / E", "Orbit Up/Down"),
        ("W / S", "Move Closer/Farther"),
        ("V / F", "Increase/Decrease Spin"),
        ("Z / X", "Decrease/Increase Mass"),
        ("T / Y", "Decrease/Increase Accretion"),
        ("G", "Toggle Grid"),
        ("C", "Science Overlay"),
        ("H", "Hide UI"),
        ("R", "Reset"),
        ("ESC", "Quit"),
    ]
    y = 458
    for k, v in controls:
        draw_text(screen, font, k, LEFT_X + 14, y, UI_TEXT)
        draw_text(screen, small_font, v, LEFT_X + 100, y + 1, UI_TEXT)
        y += 25

    # Right side: second-reference layout and neutral controls.
    draw_panel(screen, (RIGHT_X, 52, PANEL_W, 242), "Visualization")
    items = [
        ("show_disk", "Accretion Disk", state.show_disk),
        ("show_ring", "Photon Ring", state.show_ring),
        ("show_grid", "Spacetime Grid", state.show_grid),
        ("show_stars", "Stars", state.show_stars),
        ("show_particles", "Infalling Particles", state.show_particles),
    ]
    for i, (_, label, val) in enumerate(items):
        draw_checkbox(screen, font, label, val, CHECK_X, 96 + i * 28)
    draw_text(screen, font, "Color Map", SLIDER_X, 242, UI_TEXT)
    pygame.draw.rect(screen, (38, 38, 40), (SLIDER_X, 264, SLIDER_W, 30), border_radius=5)
    pygame.draw.rect(screen, (105, 108, 116, 72), (SLIDER_X, 264, SLIDER_W, 30), 1, border_radius=5)
    draw_text(screen, font, "Realistic", SLIDER_X + 10, 271, UI_TEXT)
    draw_text(screen, font, "v", SLIDER_X + SLIDER_W - 20, 270, UI_TEXT)

    draw_panel(screen, (RIGHT_X, 310, PANEL_W, 292), "Parameters")
    draw_slider(screen, font, "BH Mass (M_sun)", state.mass, SLIDER_X, 354, 4.0, 20.0, SLIDER_W)
    draw_slider(screen, font, "Spin (a)", state.spin, SLIDER_X, 406, 0.0, 1.0, SLIDER_W)
    draw_slider(screen, font, "Accretion Rate", state.accretion, SLIDER_X, 458, 0.0, 1.5, SLIDER_W)
    draw_slider(screen, font, "Time Scale", state.time_scale, SLIDER_X, 510, 0.1, 3.0, SLIDER_W)
    draw_button(screen, font, "Reset to Default", (SLIDER_X, 560, SLIDER_W, 30))

    draw_panel(screen, (RIGHT_X, 618, PANEL_W, 206), "Camera Presets")
    buttons = [
        ("Face On", 656),
        ("Edge On", 690),
        ("Top Down", 724),
        ("Close Up", 758),
        ("Far View", 792),
    ]
    for label, by in buttons:
        draw_button(screen, font, label, (SLIDER_X, by, SLIDER_W, 28))

    # Second-reference-style unobtrusive time readout.
    draw_text(screen, font, f"Time: {sim_time:5.1f} s", 14, HEIGHT - 28, UI_TEXT)
    if state.science_overlay:
        draw_text(screen, font, "Science calibration overlay: ISCO / horizon / photon critical curve", 180, HEIGHT - 28, UI_MUTED)

def reset(state):
    state.mass = 9.87
    state.spin = 0.85
    state.accretion = 0.82
    state.time_scale = 1.0
    state.yaw = 0.0
    state.pitch = 0.16
    state.distance = 5.8
    state.exposure = 1.30
    state.bloom = 1.55
    state.contrast = 1.06
    state.saturation = 0.94


def set_camera_preset(state, name):
    if name == "Face On":
        state.yaw = 0.0
        state.pitch = 0.16
        state.distance = 5.8
    elif name == "Edge On":
        state.yaw = 0.0
        state.pitch = 0.36
        state.distance = 5.9
    elif name == "Top Down":
        state.pitch = 1.12
        state.distance = 6.8
    elif name == "Close Up":
        state.pitch = 0.12
        state.distance = 3.8
    elif name == "Far View":
        state.pitch = 0.16
        state.distance = 7.6


def set_slider(state, name, mx):
    table = {
        "mass": (SLIDER_X, SLIDER_W, 4.0, 20.0),
        "spin": (SLIDER_X, SLIDER_W, 0.0, 1.0),
        "accretion": (SLIDER_X, SLIDER_W, 0.0, 1.5),
        "time_scale": (SLIDER_X, SLIDER_W, 0.1, 3.0),
    }

    sx, sw, mn, mxv = table[name]
    f = clamp((mx - sx) / sw, 0.0, 1.0)
    setattr(state, name, mn + f * (mxv - mn))


def handle_click(state, pos):
    x, y = pos

    checks = [
        ("show_disk", CHECK_X, 96),
        ("show_ring", CHECK_X, 124),
        ("show_grid", CHECK_X, 152),
        ("show_stars", CHECK_X, 180),
        ("show_particles", CHECK_X, 208),
    ]

    for name, cx, cy in checks:
        if cx <= x <= cx + 180 and cy - 4 <= y <= cy + 22:
            setattr(state, name, not getattr(state, name))
            return

    sliders = [
        ("mass", SLIDER_X, 384),
        ("spin", SLIDER_X, 436),
        ("accretion", SLIDER_X, 488),
        ("time_scale", SLIDER_X, 540),
    ]

    for name, sx, sy in sliders:
        if sx <= x <= sx + SLIDER_W and sy - 10 <= y <= sy + 15:
            state.dragging_slider = name
            set_slider(state, name, x)
            return

    if SLIDER_X <= x <= SLIDER_X + SLIDER_W and 560 <= y <= 590:
        reset(state)
        return

    presets = [
        ("Face On", 656),
        ("Edge On", 690),
        ("Top Down", 724),
        ("Close Up", 758),
        ("Far View", 792),
    ]
    for name, by in presets:
        if SLIDER_X <= x <= SLIDER_X + SLIDER_W and by <= y <= by + 28:
            set_camera_preset(state, name)
            return

    # Prevent accidental orbit drag when interacting with side panels.
    if x < 260 or x > 1284:
        return
    state.dragging = True

def main():
    pygame.init()

    # Load and downscale/cache the large NASA sky map before creating the OpenGL window.
    # This prevents the user-facing window from sitting as a black rectangle while a
    # 16K EXR is decoded. Watch the terminal for the "Loaded sky texture" message.
    sky_path = find_sky_texture()
    if sky_path == FALLBACK_SKY_PATH and not sky_path.exists():
        create_generated_milky_way(sky_path)
    sky_surface = load_quality_sky_surface(sky_path)

    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK,
        pygame.GL_CONTEXT_PROFILE_CORE,
    )

    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Real-Time Black Hole Simulator")

    ctx = moderngl.create_context()
    ctx.disable(moderngl.DEPTH_TEST)

    program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

    sky_texture = ctx.texture(sky_surface.get_size(), 3, pygame.image.tostring(sky_surface, "RGB", True))
    sky_texture.repeat_x = True
    sky_texture.repeat_y = False
    sky_texture.build_mipmaps()
    sky_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    try:
        sky_texture.anisotropy = min(8.0, ctx.info.get("GL_MAX_TEXTURE_MAX_ANISOTROPY", 1.0))
    except Exception:
        pass

    vertices = np.array(
        [
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
        ],
        dtype="f4",
    )

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(program, vbo, "in_pos")

    ui_program = ctx.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader="""
        #version 330
        uniform sampler2D u_texture;
        in vec2 v_uv;
        out vec4 fragColor;
        void main() {
            fragColor = texture(u_texture, v_uv);
        }
        """,
    )

    ui_vao = ctx.simple_vertex_array(ui_program, vbo, "in_pos")
    ui_texture = ctx.texture((WIDTH, HEIGHT), 4)

    ui_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    state = State()

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 14)
    small_font = pygame.font.SysFont("arial", 13)

    t = 0.0
    running = True
    last_mouse = None

    while running:
        dt = clock.tick(FPS) / 60.0
        fps = clock.get_fps()
        t += dt * state.time_scale

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:
                    state.show_grid = not state.show_grid
                elif event.key == pygame.K_c:
                    state.science_overlay = not state.science_overlay
                elif event.key == pygame.K_h:
                    state.show_ui = not state.show_ui
                elif event.key == pygame.K_r:
                    reset(state)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    handle_click(state, event.pos)
                    last_mouse = event.pos

                elif event.button == 4:
                    state.distance = clamp(state.distance * 0.92, 2.2, 8.0)

                elif event.button == 5:
                    state.distance = clamp(state.distance / 0.92, 2.2, 8.0)

            elif event.type == pygame.MOUSEBUTTONUP:
                state.dragging = False
                state.dragging_slider = None
                last_mouse = None

            elif event.type == pygame.MOUSEMOTION:
                if state.dragging_slider:
                    set_slider(state, state.dragging_slider, event.pos[0])

                elif state.dragging and last_mouse:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]

                    state.yaw += dx * 0.006
                    state.pitch = clamp(state.pitch + dy * 0.004, -1.25, 1.25)

                    last_mouse = event.pos

        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            state.yaw -= 0.025
        if keys[pygame.K_d]:
            state.yaw += 0.025
        if keys[pygame.K_q]:
            state.pitch = clamp(state.pitch + 0.018, -1.25, 1.25)
        if keys[pygame.K_e]:
            state.pitch = clamp(state.pitch - 0.018, -1.25, 1.25)
        if keys[pygame.K_w]:
            state.distance = clamp(state.distance * 0.985, 2.2, 8.0)
        if keys[pygame.K_s]:
            state.distance = clamp(state.distance / 0.985, 2.2, 8.0)
        if keys[pygame.K_v]:
            state.spin = clamp(state.spin + 0.008, 0.0, 1.0)
        if keys[pygame.K_f]:
            state.spin = clamp(state.spin - 0.008, 0.0, 1.0)
        if keys[pygame.K_z]:
            state.mass = clamp(state.mass - 0.06, 4.0, 20.0)
        if keys[pygame.K_x]:
            state.mass = clamp(state.mass + 0.06, 4.0, 20.0)
        if keys[pygame.K_t]:
            state.accretion = clamp(state.accretion - 0.006, 0.0, 1.5)
        if keys[pygame.K_y]:
            state.accretion = clamp(state.accretion + 0.006, 0.0, 1.5)

        program["u_resolution"].value = (WIDTH, HEIGHT)
        program["u_time"].value = t
        program["u_mass"].value = state.mass
        program["u_spin"].value = state.spin
        program["u_isco"].value = kerr_isco_radius(state.spin)
        program["u_horizon_kerr"].value = kerr_horizon_radius(state.spin)
        program["u_accretion"].value = state.accretion
        program["u_yaw"].value = state.yaw
        program["u_pitch"].value = state.pitch
        program["u_distance"].value = state.distance
        program["u_show_disk"].value = 1.0 if state.show_disk else 0.0
        program["u_show_ring"].value = 1.0 if state.show_ring else 0.0
        program["u_show_grid"].value = 1.0 if state.show_grid else 0.0
        program["u_show_stars"].value = 1.0 if state.show_stars else 0.0
        program["u_show_particles"].value = 1.0 if state.show_particles else 0.0
        program["u_exposure"].value = state.exposure
        program["u_contrast"].value = state.contrast
        program["u_saturation"].value = state.saturation
        program["u_bloom"].value = state.bloom
        program["u_science_overlay"].value = 1.0 if state.science_overlay else 0.0
        sky_texture.use(1)
        program["u_background"].value = 1

        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render()

        if state.show_ui:
            ui_surface.fill((0, 0, 0, 0))
            draw_ui(ui_surface, font, small_font, state, fps, t)

            ui_data = pygame.image.tostring(ui_surface, "RGBA", True)
            ui_texture.write(ui_data)
            ui_texture.use(0)
            ui_program["u_texture"].value = 0

            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ui_vao.render()
            ctx.disable(moderngl.BLEND)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()