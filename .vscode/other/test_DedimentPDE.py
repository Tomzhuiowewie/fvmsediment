import torch
import other.DedimentPDE as p


class MiniPINN(torch.nn.Module):
    def __init__(self, num_grain_classes=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 4 + 2 * num_grain_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    num_grain_classes = 2   # 这里假设有两种粒径类别
    model = MiniPINN(num_grain_classes=num_grain_classes)

    sediment = p.SedimentParams(grain_diameters=[0.0002, 0.002])
    fluid = p.FluidParams(manning_n=0.03)

    xyt = torch.rand(16, 3, requires_grad=True)

    residuals = p.compute_all_residuals(
        xyt=xyt,
        model=model,
        sediment=sediment,
        fluid=fluid,
        hydro_equation="swe",
        transport_method="wu",
        fall_velocity_method="soulsby",
    )

    loss = p.physics_loss(residuals)

    loss.backward()

    print("DedimentPDE 测试成功")
    print("loss =", float(loss.detach()))
    print("h:", residuals["h"].shape)
    print("C:", residuals["C"].shape)
    print("R_C:", residuals["R_C"].shape)
    print("R_z:", residuals["R_z"].shape)
    print("R_p:", residuals["R_p"].shape)


if __name__ == "__main__":
    main()