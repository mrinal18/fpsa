"""Verify the REAL fused optimizer before spending time on a full ARC run."""
import json
import torch


def check_cuda_optimizer():
    if not torch.cuda.is_available():
        raise RuntimeError('Full ARC training requires CUDA; use experiments.arc.smoke for CPU tests')
    try:
        from adam_atan2 import AdamATan2
        import adam_atan2_backend
    except (ImportError, OSError) as error:
        raise RuntimeError(
            'The official AdamATan2 CUDA extension is not usable. A Python-only '
            'wheel is insufficient. With PyTorch and CUDA headers/nvcc installed, run: '
            'python -m pip install --no-build-isolation --no-cache-dir --force-reinstall '
            'adam-atan2==0.0.3. No substitute optimizer will be selected automatically.'
        ) from error
    parameter = torch.nn.Parameter(torch.ones(8, device='cuda'))
    optimizer = AdamATan2([parameter], lr=1e-4, weight_decay=.1, betas=(.9, .95))
    parameter.square().sum().backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(parameter).all():
        raise FloatingPointError('Fused optimizer preflight produced non-finite parameters')
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), optimizer_backend=adam_atan2_backend.__file__)


if __name__ == '__main__':
    print(json.dumps(check_cuda_optimizer(), indent=2))
