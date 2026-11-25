"""Optimize and render FACET appearance stylization with the current OptiX ABI."""
import os
import sys
import torch

source_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.environ.get(
    "FACET_RENDERER_DIRECTORY", os.path.join(source_dir, "renderer", "output")
)
if output_path not in sys.path:
    sys.path.insert(0, output_path)


def packed_weights(checkpoint):
    return (
        torch.cat([checkpoint["w1_uv"], checkpoint["w1_v"], checkpoint["w1_conditioning"]], 1).to(torch.float16),
        checkpoint["w2"].to(torch.float16),
        checkpoint["w3"].to(torch.float16),
    )


def reduce_backward_weight_grads(dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3):
    dL_dw3 = dL_dw3.sum(0).reshape((8, 2, 32)).transpose(1, 2)
    dL_dw3 = dL_dw3.reshape((8, 8, -1)).transpose(0, 1).flatten(1, 2)

    dL_db3 = dL_db3.sum(0).reshape((2, 32)).transpose(0, 1)
    dL_db3 = dL_db3.reshape((8, 8)).sum(1)

    dL_dw2 = dL_dw2.sum(0).reshape((64, 2, 32)).transpose(1, 2)
    dL_dw2 = dL_dw2.reshape((8, 8, 8, -1)).transpose(1, 2).flatten(0, 1).flatten(1, 2)

    dL_db2 = dL_db2.sum(0).reshape((8, 2, 32)).transpose(1, 2)
    dL_db2 = dL_db2.reshape((8, 8, -1)).sum(2).flatten(0, 1)

    dL_dw1 = dL_dw1.sum(0).reshape((128, 2, 32)).transpose(1, 2)
    dL_dw1 = dL_dw1.reshape((8, 16, 8, -1)).transpose(1, 2).flatten(0, 1).flatten(1, 2)

    dL_db1 = dL_db1.sum(0).reshape((8, 2, 32)).transpose(1, 2)
    dL_db1 = dL_db1.reshape((8, 8, -1)).sum(2).flatten(0, 1)
    return dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3


def backward_model(
    renderer,
    checkpoint,
    conditioning_fp16,
    camera_cache,
    img_unclamped,
    img_for_loss,
    dloss_dimage,
    args,
    width,
    height,
):
    w1_fp16, w2_fp16, w3_fp16 = packed_weights(checkpoint)
    (O_chunk, v_chunk, depth_accums, reg_depth_a, reg_depth_b,
     depth_and_index, surface_normal, normal_accums) = camera_cache
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    dL_dRGB = torch.zeros_like(checkpoint.get("RGB", checkpoint["m"]))
    dL_dA = torch.zeros_like(checkpoint["A"])
    dL_dk = torch.zeros_like(checkpoint["k"])
    dL_dw3 = torch.zeros((4 * sm_count, 8 * 64), dtype=torch.float32, device="cuda")
    dL_db3 = torch.zeros((4 * sm_count, 8 * 8), dtype=torch.float32, device="cuda")
    dL_dw2 = torch.zeros((4 * sm_count, 64 * 64), dtype=torch.float32, device="cuda")
    dL_db2 = torch.zeros((4 * sm_count, 64 * 8), dtype=torch.float32, device="cuda")
    dL_dw1 = torch.zeros((4 * sm_count, 64 * 128), dtype=torch.float32, device="cuda")
    dL_db1 = torch.zeros((4 * sm_count, 64 * 8), dtype=torch.float32, device="cuda")
    # The native renderer always writes one 96D gradient row per Gaussian.
    # Planar checkpoints may store a compact N x D tensor, while the adapter
    # passes a materialized N x 96 conditioning view here.
    dL_d_conditioning = torch.zeros(
        (checkpoint["m"].shape[0], conditioning_fp16.shape[1]),
        dtype=checkpoint["conditioning_variable"].dtype,
        device=checkpoint["conditioning_variable"].device,
    )
    dL_d_features = torch.zeros_like(checkpoint["features"], dtype=torch.float32, device="cuda")
    dL_dm = torch.zeros_like(checkpoint["m"])
    dL_ds = torch.zeros_like(checkpoint["s"])
    dL_dq = torch.zeros_like(checkpoint["q"])
    depth_prefix = torch.zeros((width * height, 4), dtype=torch.float32, device="cuda")

    finite = torch.isfinite(img_unclamped)
    clamped = (~finite) | (img_unclamped != img_for_loss)
    renderer.Backward(
        conditioning_fp16,
        checkpoint["features"],
        w1_fp16,
        checkpoint["b1"],
        w2_fp16,
        checkpoint["b2"],
        w3_fp16,
        checkpoint["b3"],
        O_chunk,
        v_chunk,
        args.bg_color_R,
        args.bg_color_G,
        args.bg_color_B,
        checkpoint["m"],
        checkpoint["s"],
        checkpoint["q"],
        checkpoint["RGBA"],
        checkpoint["k"],
        img_for_loss,
        dloss_dimage * (~clamped),
        dL_dRGB,
        dL_dA,
        dL_dk,
        dL_dw3,
        dL_db3,
        dL_dw2,
        dL_db2,
        dL_dw1,
        dL_db1,
        dL_d_conditioning,
        dL_d_features,
        dL_dm,
        dL_ds,
        dL_dq,
        args.ray_termination_T_threshold_training,
        depth_accums,
        depth_prefix,
        0.0,
        reg_depth_a,
        reg_depth_b,
        depth_and_index,
        surface_normal,
        normal_accums,
        0.0,
    )
    dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3 = reduce_backward_weight_grads(
        dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3
    )
    return {
        "RGB": dL_dRGB,
        "w1_uv": dL_dw1[:, 0:8],
        "w1_v": dL_dw1[:, 8:32],
        "w1_conditioning": dL_dw1[:, 32:128],
        "b1": dL_db1,
        "w2": dL_dw2,
        "b2": dL_db2,
        "w3": torch.nn.functional.pad(dL_dw3, (0, 0, 0, 8), "constant", 0.0),
        "b3": torch.nn.functional.pad(dL_db3, (0, 8), "constant", 0.0),
        "features": dL_d_features,
        "conditioning_variable": dL_d_conditioning,
        "A": dL_dA,
        "k": dL_dk,
        "m": dL_dm,
        "s": dL_ds,
        "q": dL_dq,
    }

