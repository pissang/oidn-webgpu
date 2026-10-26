//! Deliberately narrow VkDeviceCreateInfo validator for the pinned HAL.
//! Accepts canonical individual feature structs returned by HAL, not aggregate
//! Vulkan11/12/13 aliases. Unknown REQUIRED structures fail closed.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
use ash::vk;
use std::collections::{BTreeMap, HashSet};
pub type Features = BTreeMap<String, Vec<u32>>;

// All these Vulkan structures consist of the common header followed by Bool32
// fields. Reading up through the last named Bool32 excludes trailing padding.
unsafe fn fields(base: *const vk::BaseInStructure<'_>) -> Result<Vec<u32>> {
    macro_rules! feature {
        ($ty:ident, $last:ident) => {
            if unsafe { (*base).s_type } == vk::$ty::default().s_type {
                let first = std::mem::size_of::<vk::BaseInStructure>();
                let end = std::mem::offset_of!(vk::$ty, $last) + 4;
                return Ok(unsafe {
                    std::slice::from_raw_parts(
                        base.cast::<u8>().add(first).cast::<u32>(),
                        (end - first) / 4,
                    )
                }
                .to_vec());
            }
        };
    }
    feature!(PhysicalDevice16BitStorageFeatures, storage_input_output16);
    feature!(
        PhysicalDeviceAccelerationStructureFeaturesKHR,
        descriptor_binding_acceleration_structure_update_after_bind
    );
    feature!(
        PhysicalDeviceBufferDeviceAddressFeatures,
        buffer_device_address_multi_device
    );
    feature!(
        PhysicalDeviceCooperativeMatrixFeaturesKHR,
        cooperative_matrix_robust_buffer_access
    );
    feature!(
        PhysicalDeviceDescriptorIndexingFeatures,
        runtime_descriptor_array
    );
    feature!(
        PhysicalDeviceFragmentShaderBarycentricFeaturesKHR,
        fragment_shader_barycentric
    );
    feature!(PhysicalDeviceImageRobustnessFeatures, robust_image_access);
    feature!(PhysicalDeviceMaintenance4Features, maintenance4);
    feature!(PhysicalDeviceMeshShaderFeaturesEXT, mesh_shader_queries);
    feature!(
        PhysicalDeviceMultiviewFeatures,
        multiview_tessellation_shader
    );
    feature!(
        PhysicalDevicePortabilitySubsetFeaturesKHR,
        vertex_attribute_access_beyond_stride
    );
    feature!(PhysicalDeviceRayQueryFeaturesKHR, ray_query);
    feature!(
        PhysicalDeviceRayTracingPipelineFeaturesKHR,
        ray_traversal_primitive_culling
    );
    feature!(
        PhysicalDeviceRayTracingPositionFetchFeaturesKHR,
        ray_tracing_position_fetch
    );
    feature!(PhysicalDeviceRobustness2FeaturesEXT, null_descriptor);
    feature!(
        PhysicalDeviceSamplerYcbcrConversionFeatures,
        sampler_ycbcr_conversion
    );
    feature!(
        PhysicalDeviceShaderAtomicFloatFeaturesEXT,
        sparse_image_float32_atomic_add
    );
    feature!(
        PhysicalDeviceShaderAtomicInt64Features,
        shader_shared_int64_atomics
    );
    feature!(
        PhysicalDeviceShaderDrawParametersFeatures,
        shader_draw_parameters
    );
    feature!(PhysicalDeviceShaderFloat16Int8Features, shader_int8);
    feature!(
        PhysicalDeviceShaderImageAtomicInt64FeaturesEXT,
        sparse_image_int64_atomics
    );
    feature!(
        PhysicalDeviceShaderIntegerDotProductFeatures,
        shader_integer_dot_product
    );
    feature!(
        PhysicalDeviceSubgroupSizeControlFeatures,
        compute_full_subgroups
    );
    feature!(
        PhysicalDeviceTextureCompressionASTCHDRFeatures,
        texture_compression_astc_hdr
    );
    feature!(PhysicalDeviceTimelineSemaphoreFeatures, timeline_semaphore);
    feature!(
        PhysicalDeviceVulkanMemoryModelFeatures,
        vulkan_memory_model_availability_visibility_chains
    );
    feature!(
        PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures,
        shader_zero_initialize_workgroup_memory
    );
    Err(format!(
        "unsupported feature declaration {:?}; supply the canonical HAL query chain",
        unsafe { (*base).s_type }
    )
    .into())
}
pub unsafe fn snapshot(create: &vk::DeviceCreateInfo<'_>) -> Result<Features> {
    unsafe {
        let mut result = Features::new();
        if !create.p_enabled_features.is_null() {
            if !create.p_enabled_features.is_aligned() {
                return Err("unaligned core feature declaration".into());
            }
            // VkPhysicalDeviceFeatures contains only Bool32, with no header.
            let count = std::mem::size_of::<vk::PhysicalDeviceFeatures>() / 4;
            result.insert(
                "core".into(),
                std::slice::from_raw_parts(create.p_enabled_features.cast::<u32>(), count).to_vec(),
            );
        }
        let mut next = create.p_next.cast::<vk::BaseInStructure>();
        let mut seen = HashSet::new();
        while !next.is_null() {
            if !next.is_aligned() {
                return Err("unaligned feature chain".into());
            }
            if !seen.insert((*next).s_type) {
                return Err("duplicate/cyclic feature chain".into());
            }
            if seen.len() > 64 {
                return Err("feature chain too long".into());
            }
            result.insert(format!("{:?}", (*next).s_type), fields(next)?);
            next = (*next).p_next;
        }
        Ok(result)
    }
}
pub fn validate(required: &Features, actual: &Features) -> Result<()> {
    for (name, bits) in required {
        for (index, &bit) in bits.iter().enumerate() {
            if bit != 0
                && actual
                    .get(name)
                    .and_then(|v| v.get(index))
                    .copied()
                    .unwrap_or(0)
                    == 0
            {
                return Err(format!("host did not enable HAL-required {name}[{index}]").into());
            }
        }
    }
    Ok(())
}
/// `create` and its pointed-to arrays/chain must still be valid host declarations.
pub unsafe fn validate_create(
    required: &Features,
    extensions: &[&std::ffi::CStr],
    create: &vk::DeviceCreateInfo<'_>,
    family: u32,
    queue: u32,
) -> Result<()> {
    unsafe {
        validate(required, &snapshot(create)?)?;
        let actual = if create.enabled_extension_count == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(
                create.pp_enabled_extension_names,
                create.enabled_extension_count as usize,
            )
        };
        for &extension in extensions {
            if !actual
                .iter()
                .any(|p| !p.is_null() && std::ffi::CStr::from_ptr(*p) == extension)
            {
                return Err(format!("missing HAL-required device extension {extension:?}").into());
            }
        }
        let queues = if create.queue_create_info_count == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(
                create.p_queue_create_infos,
                create.queue_create_info_count as usize,
            )
        };
        if !queues
            .iter()
            .any(|q| q.queue_family_index == family && q.queue_count > queue && q.flags.is_empty())
        {
            return Err("host did not create the requested unprotected queue".into());
        }
        Ok(())
    }
}
#[allow(dead_code)]
pub fn negative_checks(required: &Features) -> Result<usize> {
    let mut checked = 0;
    for (name, bits) in required {
        for (index, &bit) in bits.iter().enumerate() {
            if bit == 0 {
                continue;
            }
            let mut missing = required.clone();
            missing.get_mut(name).unwrap()[index] = 0;
            if validate(required, &missing).is_ok() {
                return Err("feature rejection failed".into());
            }
            checked += 1;
        }
    }
    Ok(checked)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn import_requires_actual_enabled_features_extensions_and_queue() {
        unsafe {
            let core = vk::PhysicalDeviceFeatures::default().robust_buffer_access(true);
            let mut timeline =
                vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
            let mut robustness =
                vk::PhysicalDeviceRobustness2FeaturesEXT::default().robust_buffer_access2(true);
            let names = [c"VK_EXT_robustness2".as_ptr()];
            let priorities = [1.0, 1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(3)
                .queue_priorities(&priorities)];
            let create = vk::DeviceCreateInfo::default()
                .enabled_features(&core)
                .push_next(&mut timeline)
                .push_next(&mut robustness)
                .enabled_extension_names(&names)
                .queue_create_infos(&queues);
            let required = snapshot(&create).unwrap();
            validate_create(&required, &[c"VK_EXT_robustness2"], &create, 3, 1).unwrap();
            assert_eq!(negative_checks(&required).unwrap(), 3);
            assert!(validate_create(&required, &[c"missing"], &create, 3, 1).is_err());
            assert!(validate_create(&required, &[], &create, 2, 1).is_err());
            assert!(validate_create(&required, &[], &create, 3, 2).is_err());
            let missing = vk::DeviceCreateInfo {
                p_enabled_features: std::ptr::null(),
                ..create
            };
            assert!(validate_create(&required, &[], &missing, 3, 1).is_err());
            let mut aggregate =
                vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
            assert!(snapshot(&vk::DeviceCreateInfo::default().push_next(&mut aggregate)).is_err());
        }
    }
}

pub unsafe fn cooperative_matrix_enabled(create: &vk::DeviceCreateInfo<'_>) -> Result<bool> {
    unsafe {
        let actual = snapshot(create)?;
        let matrix = actual.get("PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR")
            .and_then(|bits| bits.first()).copied() == Some(1);
        if create.enabled_extension_count > 4096 {
            return Err("extension count exceeds 4096 entries".into());
        }
        if create.enabled_extension_count == 0 { return Ok(false); }
        if create.pp_enabled_extension_names.is_null() || !create.pp_enabled_extension_names.is_aligned() {
            return Err("invalid extension array".into());
        }
        let names = std::slice::from_raw_parts(create.pp_enabled_extension_names, create.enabled_extension_count as usize);
        Ok(matrix && names.iter().any(|p| !p.is_null() && std::ffi::CStr::from_ptr(*p) == ash::khr::cooperative_matrix::NAME))
    }
}

#[cfg(test)]
#[test]
fn matrix_import_requires_both_extension_and_feature() {
    let names = [ash::khr::cooperative_matrix::NAME.as_ptr()];
    let mut matrix = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
    let enabled = vk::DeviceCreateInfo::default().enabled_extension_names(&names).push_next(&mut matrix);
    unsafe {
        assert!(cooperative_matrix_enabled(&enabled).unwrap());
        assert!(!cooperative_matrix_enabled(&vk::DeviceCreateInfo::default().enabled_extension_names(&names)).unwrap());
        assert!(!cooperative_matrix_enabled(&vk::DeviceCreateInfo::default().push_next(&mut matrix)).unwrap());
    }
}
