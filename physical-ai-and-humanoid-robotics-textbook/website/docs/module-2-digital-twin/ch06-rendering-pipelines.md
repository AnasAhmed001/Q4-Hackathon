---
title: Chapter 6 - High-Fidelity Rendering Pipelines
description: Explore high-fidelity rendering techniques in Unity including PBR, HDR, and advanced lighting for creating realistic humanoid robot simulations.
sidebar_position: 18
---

# Chapter 6 - High-Fidelity Rendering Pipelines

Creating realistic visual environments is crucial for digital twins, especially when training humanoid robots that rely on visual perception. High-fidelity rendering in Unity involves using advanced rendering techniques such as Physically Based Rendering (PBR), High Dynamic Range (HDR) lighting, and advanced shading to create photorealistic scenes that closely match real-world conditions.

## 6.1 Introduction to Rendering Pipelines in Unity

Unity provides several rendering pipelines optimized for different use cases:

- **Built-in Render Pipeline**: Unity's original rendering system, flexible but less optimized
- **Universal Render Pipeline (URP)**: Lightweight and efficient, suitable for multi-platform projects
- **High Definition Render Pipeline (HDRP)**: High-fidelity rendering for desktop and console platforms
- **Scriptable Render Pipeline (SRP)**: Customizable rendering pipeline for specific needs

For humanoid robot digital twins requiring high-fidelity visuals, HDRP is often the best choice, though URP can provide good results with better performance.

## 6.2 Physically Based Rendering (PBR)

PBR is a rendering approach that simulates light interaction with surfaces based on physical principles, resulting in more realistic materials and lighting.

### 6.2.1 PBR Material Properties

PBR materials in Unity typically use two workflows:

**Metallic-Roughness Workflow:**
- **Albedo (Base Color)**: The base color of the material
- **Metallic**: Defines if the surface is metallic (1) or non-metallic (0)
- **Smoothness/Roughness**: Controls surface smoothness (inverse of roughness)
- **Normal Map**: Simulates surface detail without geometry
- **Occlusion**: Simulates ambient light occlusion
- **Height Map**: Adds displacement effects

**Specular-Glossiness Workflow:**
- **Albedo**: Base color
- **Specular**: Color of reflections
- **Glossiness**: Surface smoothness
- **Normal Map**: Surface detail simulation

### 6.2.2 Creating PBR Materials for Humanoid Robots

Here's an example of creating a metallic humanoid robot material in Unity:

```csharp
// Material creation script for PBR materials
using UnityEngine;

public class HumanoidMaterialSetup : MonoBehaviour
{
    [Header("Material Properties")]
    public Material robotBodyMaterial;
    public Texture2D albedoMap;
    public Texture2D metallicMap;
    public Texture2D normalMap;
    public Texture2D roughnessMap;

    void Start()
    {
        if (robotBodyMaterial != null)
        {
            SetupPBRMaterial(robotBodyMaterial);
        }
    }

    void SetupPBRMaterial(Material material)
    {
        // Set up PBR properties
        if (albedoMap != null)
            material.SetTexture("_BaseMap", albedoMap);

        if (metallicMap != null)
            material.SetTexture("_MetallicGlossMap", metallicMap);

        if (normalMap != null)
            material.SetTexture("_BumpMap", normalMap);

        if (roughnessMap != null)
            material.SetTexture("_SpecGlossMap", roughnessMap);

        // Additional properties
        material.SetFloat("_Metallic", 0.8f); // Metallic surface
        material.SetFloat("_Smoothness", 0.6f); // Smooth surface
        material.SetFloat("_BumpScale", 1.0f); // Normal map intensity
    }
}
```

## 6.3 High Dynamic Range (HDR) Lighting

HDR lighting allows for a wider range of luminance values, creating more realistic lighting conditions that match real-world scenarios.

### 6.3.1 Setting up HDR in Unity

To enable HDR in your Unity project:

1. In the Universal Render Pipeline Asset or HDRP Asset, enable HDR
2. Configure your camera to support HDR
3. Use HDR colors and textures for lighting

```csharp
// HDR Camera Configuration
using UnityEngine;
using UnityEngine.Rendering;

public class HDRCameraSetup : MonoBehaviour
{
    void Start()
    {
        Camera camera = GetComponent<Camera>();

        // Enable HDR (if using URP/HDRP)
        camera.allowHDR = true;
        camera.allowMSAA = true;
        camera.allowDynamicResolution = true;
    }
}
```

### 6.3.2 HDR Lighting Setup

```csharp
// HDR Environment Lighting
using UnityEngine;

public class HDRLightingSetup : MonoBehaviour
{
    [Header("HDR Settings")]
    public Light sunLight;
    public float sunIntensity = 100000f; // Lux for sun
    public Color sunColor = new Color(1f, 0.95f, 0.8f, 1f); // Warm sunlight
    public Cubemap skyboxCubemap;

    void Start()
    {
        SetupHDR();
    }

    void SetupHDR()
    {
        if (sunLight != null)
        {
            // Set high intensity for realistic sunlight
            sunLight.intensity = sunIntensity;
            sunLight.color = sunColor;
            sunLight.useColorTemperature = true;
            sunLight.colorTemperature = 6500f; // Daylight temperature
        }

        // Configure skybox for HDR
        if (skyboxCubemap != null)
        {
            RenderSettings.skybox.SetTexture("_Tex", skyboxCubemap);
        }
    }
}
```

## 6.4 Advanced Lighting Techniques

### 6.4.1 Real-time Global Illumination

Global illumination simulates how light bounces off surfaces, creating more realistic indirect lighting:

```csharp
// Global Illumination Setup
using UnityEngine;
using UnityEngine.Rendering;

public class GlobalIlluminationSetup : MonoBehaviour
{
    [Header("Lighting Settings")]
    public Light[] areaLights;
    public float indirectIntensity = 1.5f;

    void Start()
    {
        SetupGlobalIllumination();
    }

    void SetupGlobalIllumination()
    {
        // Configure area lights for soft shadows
        foreach (Light light in areaLights)
        {
            if (light.type == LightType.Rectangle || light.type == LightType.Disc)
            {
                light.bounceIntensity = indirectIntensity;
                light.useBakery = true; // If using Bakery GI
            }
        }
    }
}
```

### 6.4.2 Image-Based Lighting (IBL)

IBL uses environment maps to simulate realistic reflections and lighting:

```csharp
// Image-Based Lighting Setup
using UnityEngine;

public class ImageBasedLighting : MonoBehaviour
{
    [Header("IBL Settings")]
    public Cubemap reflectionCubemap;
    public float reflectionIntensity = 1.0f;
    public Material[] reflectiveMaterials;

    void Start()
    {
        SetupIBL();
    }

    void SetupIBL()
    {
        // Set up reflection probe
        var reflectionProbe = GetComponent<ReflectionProbe>();
        if (reflectionProbe != null)
        {
            reflectionProbe.mode = ReflectionProbeMode.Baked;
            reflectionProbe.intensity = reflectionIntensity;
        }

        // Apply to materials
        foreach (Material mat in reflectiveMaterials)
        {
            if (mat != null)
            {
                mat.SetTexture("_ReflectionTexture", reflectionCubemap);
            }
        }
    }
}
```

## 6.5 Post-Processing Effects

Post-processing effects enhance the visual quality of the rendered scene:

```csharp
// Post-Processing Setup for Robotics Simulation
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class PostProcessingSetup : MonoBehaviour
{
    [Header("Post-Processing Settings")]
    public VolumeProfile volumeProfile;

    void Start()
    {
        SetupPostProcessing();
    }

    void SetupPostProcessing()
    {
        if (volumeProfile != null)
        {
            // Add post-processing effects
            AddBloomEffect(volumeProfile);
            AddColorGrading(volumeProfile);
            AddVignette(volumeProfile);
        }
    }

    void AddBloomEffect(VolumeProfile profile)
    {
        Bloom bloom;
        if (!profile.TryGet(out bloom))
        {
            bloom = profile.Add<Bloom>();
        }

        bloom.threshold.value = 1.0f;
        bloom.intensity.value = 0.5f;
        bloom.scatter.value = 0.7f;
    }

    void AddColorGrading(VolumeProfile profile)
    {
        ColorAdjustments colorAdjust;
        if (!profile.TryGet(out colorAdjust))
        {
            colorAdjust = profile.Add<ColorAdjustments>();
        }

        colorAdjust.postExposure.value = 0.2f;
        colorAdjust.contrast.value = 10f;
    }

    void AddVignette(VolumeProfile profile)
    {
        Vignette vignette;
        if (!profile.TryGet(out vignette))
        {
            vignette = profile.Add<Vignette>();
        }

        vignette.intensity.value = 0.2f;
        vignette.smoothness.value = 0.4f;
    }
}
```

## 6.6 Performance Optimization for Real-time Rendering

For humanoid robot simulation, maintaining high frame rates is crucial:

### 6.6.1 Level of Detail (LOD)

```csharp
// LOD Setup for Robot Models
using UnityEngine;

public class RobotLODSetup : MonoBehaviour
{
    [System.Serializable]
    public class LODGroup
    {
        public GameObject[] meshes;
        public float screenRelativeTransitionHeight;
        public float fadeTransitionWidth;
    }

    public LODGroup[] lodGroups;

    void Start()
    {
        SetupLOD();
    }

    void SetupLOD()
    {
        LODGroupSystem lodSystem = GetComponent<LODGroup>();
        if (lodSystem == null)
        {
            lodSystem = gameObject.AddComponent<LODGroup>();
        }

        LOD[] lods = new LOD[lodGroups.Length];
        for (int i = 0; i < lodGroups.Length; i++)
        {
            lods[i] = new LOD(lodGroups[i].screenRelativeTransitionHeight,
                             lodGroups[i].meshes);
            lods[i].fadeTransitionWidth = lodGroups[i].fadeTransitionWidth;
        }

        lodSystem.SetLODs(lods);
        lodSystem.RecalculateBounds();
    }
}
```

### 6.6.2 Occlusion Culling

Enable occlusion culling in Unity to avoid rendering objects not visible to the camera:

1. Mark static objects with "Occluder Static" and "Occludee Static"
2. Bake occlusion culling data in Window > Rendering > Occlusion Culling

## 6.7 Rendering for Computer Vision

For humanoid robots using computer vision, ensure the rendering pipeline produces data suitable for AI training:

```csharp
// Computer Vision Rendering Setup
using UnityEngine;
using System.Collections;

public class CVRenderingSetup : MonoBehaviour
{
    [Header("Computer Vision Settings")]
    public Camera rgbCamera;
    public Camera depthCamera;
    public RenderTexture rgbTexture;
    public RenderTexture depthTexture;
    public int imageWidth = 640;
    public int imageHeight = 480;

    void Start()
    {
        SetupCVRendering();
    }

    void SetupCVRendering()
    {
        // Configure RGB camera
        if (rgbCamera != null)
        {
            rgbTexture = new RenderTexture(imageWidth, imageHeight, 24);
            rgbCamera.targetTexture = rgbTexture;
        }

        // Configure depth camera
        if (depthCamera != null)
        {
            depthTexture = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.Depth);
            depthCamera.targetTexture = depthTexture;
        }
    }

    // Function to capture images for training data
    public Texture2D CaptureRGBImage()
    {
        RenderTexture.active = rgbCamera.targetTexture;
        Texture2D image = new Texture2D(rgbTexture.width, rgbTexture.height, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, rgbTexture.width, rgbTexture.height), 0, 0);
        image.Apply();
        RenderTexture.active = null;
        return image;
    }
}
```

## 6.8 Sensor Simulation for Computer Vision

Simulate realistic sensor properties for computer vision training:

```csharp
// Sensor Noise Simulation
using UnityEngine;
using System.Collections;

public class SensorNoiseSimulation : MonoBehaviour
{
    [Header("Noise Parameters")]
    public float gaussianNoise = 0.01f;
    public float saltPepperNoise = 0.005f;
    public float motionBlurIntensity = 0.1f;

    // Apply noise to captured images
    public Texture2D ApplyNoise(Texture2D image)
    {
        Color[] pixels = image.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            // Apply Gaussian noise
            if (Random.value < gaussianNoise)
            {
                pixels[i] += new Color(
                    Random.Range(-0.1f, 0.1f),
                    Random.Range(-0.1f, 0.1f),
                    Random.Range(-0.1f, 0.1f)
                );
            }

            // Apply salt and pepper noise
            if (Random.value < saltPepperNoise)
            {
                pixels[i] = Random.value > 0.5f ? Color.white : Color.black;
            }
        }

        image.SetPixels(pixels);
        image.Apply();
        return image;
    }
}
```

## Summary

High-fidelity rendering pipelines are essential for creating realistic digital twins of humanoid robots. By implementing PBR, HDR lighting, advanced post-processing, and computer vision-specific optimizations, you can create simulation environments that closely match real-world conditions. This enables more effective training of vision-based AI systems and better sim-to-real transfer. In the next chapter, we will explore the ROS-Unity bridge for bidirectional communication between ROS 2 and Unity.