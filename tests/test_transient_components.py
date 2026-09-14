"""Tests for transient BTE composable components."""

import pytest
import torch
import numpy as np

from pinn_bte.training.transient import (
    # Types
    BCType,
    ICType,
    HeatSourceType,
    get_transient_wandb_tags,
    # Boundary conditions
    PeriodicBC,
    IsothermalBC,
    AdiabaticBC,
    GaussianBC,
    # Initial conditions
    UniformIC,
    CosineIC,
    # Heat sources
    NoHeatSource,
    StepHeatSource,
    PulseHeatSource,
    PeriodicHeatSource,
    # Factories
    create_bc,
    create_ic,
    create_heat_source,
    create_bc_pair,
)


class TestBCType:
    """Tests for BCType enum."""

    def test_bc_type_values(self):
        """Test that BCType has expected values."""
        assert BCType.PERIODIC.value == "periodic"
        assert BCType.HOT.value == "hot"
        assert BCType.COLD.value == "cold"
        assert BCType.INSULATED.value == "insulated"
        assert BCType.GAUSSIAN.value == "gaussian"

    def test_bc_type_from_string(self):
        """Test creating BCType from string value."""
        assert BCType("periodic") == BCType.PERIODIC
        assert BCType("hot") == BCType.HOT


class TestICType:
    """Tests for ICType enum."""

    def test_ic_type_values(self):
        """Test that ICType has expected values."""
        assert ICType.UNIFORM.value == "uniform"
        assert ICType.COSINE.value == "cosine"
        assert ICType.STEADY_STATE.value == "steady_state"


class TestHeatSourceType:
    """Tests for HeatSourceType enum."""

    def test_heat_source_type_values(self):
        """Test that HeatSourceType has expected values."""
        assert HeatSourceType.NONE.value == "none"
        assert HeatSourceType.STEP.value == "step"
        assert HeatSourceType.PULSE.value == "pulse"
        assert HeatSourceType.PERIODIC_TIME.value == "periodic_time"


class TestWandbTags:
    """Tests for standardized W&B tag generation."""

    def test_basic_tags_1d(self):
        """Test basic 1D tag generation returns flat list."""
        tags = get_transient_wandb_tags(
            dimension="1d",
            model="gray",
            bc_left=BCType.PERIODIC,
            bc_right=BCType.PERIODIC,
            ic_type=ICType.COSINE,
            heat_source=HeatSourceType.NONE,
        )

        assert isinstance(tags, list)
        assert "transient" in tags
        assert "1d" in tags
        assert "gray" in tags
        assert "small_dt" in tags
        assert "ic=cosine" in tags
        assert "bc=periodic" in tags
        # HeatSourceType.NONE should not produce a source= tag
        assert not any(t.startswith("source=") for t in tags)

    def test_tags_2d_with_extra(self):
        """Test 2D tag generation with extra parameters."""
        tags = get_transient_wandb_tags(
            dimension="2d",
            model="nongray",
            bc_left=BCType.COLD,
            bc_right=BCType.COLD,
            ic_type=ICType.UNIFORM,
            heat_source=HeatSourceType.STEP,
            delta_T_regime="large_dt",
            bc_top=BCType.GAUSSIAN,
            bc_bottom=BCType.COLD,
            L_um="10.0",
            Kn="0.1",
        )

        assert isinstance(tags, list)
        assert "2d" in tags
        assert "nongray" in tags
        assert "large_dt" in tags
        assert "ic=uniform" in tags
        assert "bc=cold" in tags
        assert "bc=gaussian" in tags
        assert "source=step" in tags
        assert "L=10.0um" in tags
        assert "Kn=0.1" in tags


class TestBoundaryConditions:
    """Tests for boundary condition classes."""

    def test_periodic_bc_type(self):
        """Test PeriodicBC returns correct type."""
        bc = PeriodicBC()
        assert bc.bc_type == BCType.PERIODIC

    def test_isothermal_bc_hot(self):
        """Test IsothermalBC with hot type."""
        bc = IsothermalBC(T_target=1.0, bc_type_value=BCType.HOT)
        assert bc.bc_type == BCType.HOT
        assert bc.T_target == 1.0

    def test_isothermal_bc_cold(self):
        """Test IsothermalBC with cold type."""
        bc = IsothermalBC(T_target=0.0, bc_type_value=BCType.COLD)
        assert bc.bc_type == BCType.COLD
        assert bc.T_target == 0.0

    def test_adiabatic_bc_type(self):
        """Test AdiabaticBC returns correct type."""
        bc = AdiabaticBC()
        assert bc.bc_type == BCType.INSULATED

    def test_gaussian_bc_target(self):
        """Test GaussianBC target computation."""
        bc = GaussianBC(T_c=0.0, delta_T=1.0, x0=0.5, sigma=0.2)
        assert bc.bc_type == BCType.GAUSSIAN

        # Test target at center (x=0.5) should be T_c + delta_T = 1.0
        x_center = torch.tensor([[0.5]])
        target = bc.get_target(x_center)
        assert torch.isclose(target, torch.tensor([[1.0]]), atol=1e-6)

        # Test target at edge (x=0) should be near T_c
        x_edge = torch.tensor([[0.0]])
        target_edge = bc.get_target(x_edge)
        assert target_edge.item() < 0.1  # Should be close to 0


class TestInitialConditions:
    """Tests for initial condition classes."""

    def test_uniform_ic_type(self):
        """Test UniformIC returns correct type."""
        ic = UniformIC(T_initial=300.0)
        assert ic.ic_type == ICType.UNIFORM

    def test_uniform_ic_target(self):
        """Test UniformIC returns constant target."""
        ic = UniformIC(T_initial=0.5)
        x = torch.tensor([[0.0], [0.5], [1.0]])
        target = ic.get_T_target(x)
        expected = torch.full_like(x, 0.5)
        assert torch.allclose(target, expected)

    def test_cosine_ic_type(self):
        """Test CosineIC returns correct type."""
        ic = CosineIC(amplitude=1.0)
        assert ic.ic_type == ICType.COSINE

    def test_cosine_ic_target(self):
        """Test CosineIC returns cosine profile."""
        ic = CosineIC(amplitude=1.0, k=2 * np.pi)

        # At x=0, cos(0) = 1
        x_0 = torch.tensor([[0.0]])
        target_0 = ic.get_T_target(x_0)
        assert torch.isclose(target_0, torch.tensor([[1.0]]), atol=1e-6)

        # At x=0.5, cos(π) = -1
        x_half = torch.tensor([[0.5]])
        target_half = ic.get_T_target(x_half)
        assert torch.isclose(target_half, torch.tensor([[-1.0]]), atol=1e-6)


class TestHeatSources:
    """Tests for heat source classes."""

    def test_no_heat_source_type(self):
        """Test NoHeatSource returns correct type."""
        hs = NoHeatSource()
        assert hs.source_type == HeatSourceType.NONE

    def test_no_heat_source_power(self):
        """Test NoHeatSource returns zero power."""
        hs = NoHeatSource()
        x = torch.tensor([[0.5]])
        t = torch.tensor([[0.5]])
        power = hs.get_power(x, t)
        assert torch.isclose(power, torch.tensor([[0.0]]))

    def test_step_heat_source_type(self):
        """Test StepHeatSource returns correct type."""
        hs = StepHeatSource()
        assert hs.source_type == HeatSourceType.STEP

    def test_pulse_heat_source_type(self):
        """Test PulseHeatSource returns correct type."""
        hs = PulseHeatSource(Q0=1.0, x0=0.5, t0=0.1, sigma_x=0.1, sigma_t=0.05)
        assert hs.source_type == HeatSourceType.PULSE

    def test_pulse_heat_source_power(self):
        """Test PulseHeatSource returns Gaussian pulse."""
        hs = PulseHeatSource(Q0=1.0, x0=0.5, t0=0.1, sigma_x=0.1, sigma_t=0.05)

        x_center = torch.tensor([[0.5]])
        t_center = torch.tensor([[0.1]])
        power_center = hs.get_power(x_center, t_center)
        assert torch.isclose(power_center, torch.tensor([[1.0]]), atol=1e-6)

        # Away from center, power should be less
        x_edge = torch.tensor([[0.0]])
        t_late = torch.tensor([[0.5]])
        power_edge = hs.get_power(x_edge, t_late)
        assert power_edge.item() < 0.1

    def test_periodic_heat_source_type(self):
        """Test PeriodicHeatSource returns correct type."""
        hs = PeriodicHeatSource(Q0=1.0, omega=2 * np.pi)
        assert hs.source_type == HeatSourceType.PERIODIC_TIME


class TestFactories:
    """Tests for factory functions."""

    def test_create_bc_periodic(self):
        """Test creating PeriodicBC via factory."""
        bc = create_bc(BCType.PERIODIC)
        assert isinstance(bc, PeriodicBC)
        assert bc.bc_type == BCType.PERIODIC

    def test_create_bc_hot(self):
        """Test creating hot IsothermalBC via factory."""
        bc = create_bc(BCType.HOT, T_hot=400.0)
        assert isinstance(bc, IsothermalBC)
        assert bc.bc_type == BCType.HOT
        assert bc.T_target == 400.0

    def test_create_bc_cold(self):
        """Test creating cold IsothermalBC via factory."""
        bc = create_bc(BCType.COLD, T_cold=300.0)
        assert isinstance(bc, IsothermalBC)
        assert bc.bc_type == BCType.COLD
        assert bc.T_target == 300.0

    def test_create_bc_insulated(self):
        """Test creating AdiabaticBC via factory."""
        bc = create_bc(BCType.INSULATED)
        assert isinstance(bc, AdiabaticBC)
        assert bc.bc_type == BCType.INSULATED

    def test_create_bc_gaussian(self):
        """Test creating GaussianBC via factory."""
        bc = create_bc(BCType.GAUSSIAN, T_c=0.0, delta_T=1.0, x0=0.5, sigma=0.2)
        assert isinstance(bc, GaussianBC)
        assert bc.bc_type == BCType.GAUSSIAN
        assert bc.delta_T == 1.0

    def test_create_bc_invalid(self):
        """Test that invalid BC type raises ValueError."""
        with pytest.raises(ValueError):
            create_bc("invalid_type")

    def test_create_ic_uniform(self):
        """Test creating UniformIC via factory."""
        ic = create_ic(ICType.UNIFORM, T_initial=0.0)
        assert isinstance(ic, UniformIC)
        assert ic.ic_type == ICType.UNIFORM

    def test_create_ic_cosine(self):
        """Test creating CosineIC via factory."""
        ic = create_ic(ICType.COSINE, amplitude=2.0)
        assert isinstance(ic, CosineIC)
        assert ic.ic_type == ICType.COSINE
        assert ic.amplitude == 2.0

    def test_create_ic_invalid(self):
        """Test that invalid IC type raises ValueError."""
        with pytest.raises(ValueError):
            create_ic("invalid_type")

    def test_create_heat_source_none(self):
        """Test creating NoHeatSource via factory."""
        hs = create_heat_source(HeatSourceType.NONE)
        assert isinstance(hs, NoHeatSource)

    def test_create_heat_source_step(self):
        """Test creating StepHeatSource via factory."""
        hs = create_heat_source(HeatSourceType.STEP)
        assert isinstance(hs, StepHeatSource)

    def test_create_heat_source_pulse(self):
        """Test creating PulseHeatSource via factory."""
        hs = create_heat_source(HeatSourceType.PULSE, Q0=2.0, t0=0.2)
        assert isinstance(hs, PulseHeatSource)
        assert hs.Q0 == 2.0
        assert hs.t0 == 0.2

    def test_create_heat_source_invalid(self):
        """Test that invalid heat source type raises ValueError."""
        with pytest.raises(ValueError):
            create_heat_source("invalid_type")

    def test_create_bc_pair(self):
        """Test creating BC pair via factory."""
        bc_left, bc_right = create_bc_pair(BCType.HOT, BCType.COLD, T_hot=1.0, T_cold=0.0)
        assert bc_left.bc_type == BCType.HOT
        assert bc_right.bc_type == BCType.COLD


class TestImports:
    """Tests that all imports work correctly."""

    def test_import_from_transient(self):
        """Test that all components can be imported from transient module."""
        from pinn_bte.training.transient import (
            BCType,
            ICType,
            HeatSourceType,
            get_transient_wandb_tags,
            TransientBC,
            PeriodicBC,
            IsothermalBC,
            AdiabaticBC,
            GaussianBC,
            InitialCondition,
            UniformIC,
            CosineIC,
            HeatSource,
            NoHeatSource,
            StepHeatSource,
            PulseHeatSource,
            create_bc,
            create_ic,
            create_heat_source,
        )
        # If we get here without error, imports work
        assert True

    def test_trainers_have_bc_ic_types(self):
        """Test that trainers expose BC/IC types in their __init__."""
        from pinn_bte.training.transient import Transient1DGrayTrainer
        import inspect

        sig = inspect.signature(Transient1DGrayTrainer.__init__)
        params = sig.parameters

        assert "bc_type" in params
        assert "ic_type" in params
        assert "heat_source" in params
