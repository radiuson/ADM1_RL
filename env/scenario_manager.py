#!/usr/bin/env python3
"""
Scenario Manager for Biogas RL Benchmark

Manages scenario configurations, disturbance injection, and initial state setup.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional


class ScenarioManager:
    """
    Manages evaluation scenarios for RL benchmark

    Features:
        - Load scenarios from YAML config
        - Apply disturbances at specified times
        - Generate custom initial states
        - Track scenario progress
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize scenario manager

        Args:
            config_path: Path to scenarios.yaml (None = use default)
        """
        if config_path is None:
            # Default: same directory as this file
            config_path = Path(__file__).parent / "scenarios.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.scenarios = self.config['scenarios']
        self.initial_states = self.config['initial_states']
        self.scenario_groups = self.config['scenario_groups']

        # Current scenario tracking
        self.current_scenario = None
        self.current_time = 0.0  # days
        self.step_count = 0

    def load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Load scenario configuration

        Args:
            scenario_name: Name of scenario (e.g., 'nominal', 'shock_load')

        Returns:
            Scenario configuration dict
        """
        if scenario_name not in self.scenarios:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(self.scenarios.keys())}"
            )

        self.current_scenario = self.scenarios[scenario_name]
        self.current_time = 0.0
        self.step_count = 0

        return self.current_scenario

    def get_initial_state(self, scenario_name: str) -> Dict[str, float]:
        """
        Get initial state for scenario

        Args:
            scenario_name: Scenario name

        Returns:
            Initial state dictionary
        """
        scenario = self.scenarios[scenario_name]
        initial_state_name = scenario['initial_state']

        if initial_state_name not in self.initial_states:
            raise ValueError(f"Unknown initial state '{initial_state_name}'")

        state_config = self.initial_states[initial_state_name]

        if state_config['source'] == 'csv':
            return self._load_state_from_csv(state_config['file'])
        elif state_config['source'] == 'custom':
            # Use custom state dict
            return state_config['state']
        else:
            raise ValueError(f"Unknown state source: {state_config['source']}")

    def _load_state_from_csv(self, filename: str) -> Dict[str, float]:
        """Load initial state from CSV file"""
        # CSV files are stored alongside the env package in env/data/
        csv_path = Path(__file__).parent / 'data' / filename

        if not csv_path.exists():
            raise FileNotFoundError(f"Initial state CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        row = df.iloc[0]

        # Convert to dictionary
        state = {}
        for col in row.index:
            state[col] = row[col]

        # Ensure pH is present
        if 'pH' not in state:
            state['pH'] = 7.26

        return state

    def apply_influent_multiplier(self,
                                   base_influent: Dict[str, float],
                                   scenario_name: str) -> Dict[str, float]:
        """
        Apply influent multiplier from scenario config

        Args:
            base_influent: Base influent dict
            scenario_name: Current scenario

        Returns:
            Modified influent dict
        """
        scenario = self.scenarios[scenario_name]
        influent_config = scenario['influent']

        if influent_config['type'] == 'default':
            multiplier = influent_config['multiplier']

            # Apply multiplier to organic substrates
            modified_influent = base_influent.copy()
            for key in ['X_ch', 'X_pr', 'X_li', 'X_xc']:
                if key in modified_influent:
                    modified_influent[key] *= multiplier

            return modified_influent

        elif influent_config['type'] == 'ramp':
            # Ramp influent type: not used by current paper scenarios
            return base_influent

        else:
            return base_influent

    def check_disturbances(self,
                           current_time_days: float) -> Optional[Dict[str, Any]]:
        """
        Check if any disturbance should be applied at current time

        Args:
            current_time_days: Current simulation time (days)

        Returns:
            Disturbance dict if active, None otherwise
        """
        if self.current_scenario is None:
            return None

        disturbances = self.current_scenario.get('disturbances', [])

        for disturbance in disturbances:
            dist_time = disturbance['time_days']
            dist_type = disturbance['type']

            if dist_type == 'influent_spike':
                # Check if we're in the spike window
                duration_hours = disturbance.get('duration_hours', 2)
                duration_days = duration_hours / 24.0

                if dist_time <= current_time_days < dist_time + duration_days:
                    return disturbance

            elif dist_type == 'temperature_ramp':
                # Check if we're in the ramp period
                duration_days = disturbance.get('duration_days', 1.0)

                if dist_time <= current_time_days < dist_time + duration_days:
                    # Calculate current temperature in ramp
                    from_temp = disturbance['from_temp']
                    to_temp = disturbance['to_temp']
                    progress = (current_time_days - dist_time) / duration_days
                    current_temp = from_temp + (to_temp - from_temp) * progress

                    return {
                        'type': 'temperature_ramp',
                        'current_temp': current_temp,
                        'progress': progress
                    }

        return None

    def apply_disturbance(self,
                          base_influent: Dict[str, float],
                          disturbance: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply disturbance to influent

        Args:
            base_influent: Base influent dict
            disturbance: Disturbance configuration

        Returns:
            Modified influent dict
        """
        if disturbance['type'] == 'influent_spike':
            magnitude = disturbance['magnitude']
            affected_vars = disturbance.get('affected_vars', ['X_ch', 'X_pr', 'X_li', 'X_xc'])

            modified_influent = base_influent.copy()
            for var in affected_vars:
                if var in modified_influent:
                    modified_influent[var] *= magnitude

            return modified_influent

        # Other disturbance types don't modify influent
        return base_influent

    def get_temperature(self,
                        scenario_name: str,
                        current_time_days: float) -> float:
        """
        Get current temperature for scenario

        Args:
            scenario_name: Scenario name
            current_time_days: Current simulation time (days)

        Returns:
            Temperature in °C
        """
        scenario = self.scenarios[scenario_name]
        base_temp = scenario.get('temperature', 35.0)

        # Check for temperature disturbances
        disturbance = self.check_disturbances(current_time_days)
        if disturbance and disturbance['type'] == 'temperature_ramp':
            return disturbance['current_temp']

        return base_temp

    def get_scenario_duration(self, scenario_name: str) -> float:
        """
        Get scenario duration in days

        Args:
            scenario_name: Scenario name

        Returns:
            Duration in days
        """
        return self.scenarios[scenario_name]['duration_days']

    def get_scenario_group(self, group_name: str) -> List[str]:
        """
        Get list of scenarios in a group

        Args:
            group_name: Group name (e.g., 'mvp', 'full', 'robustness')

        Returns:
            List of scenario names
        """
        if group_name not in self.scenario_groups:
            raise ValueError(
                f"Unknown scenario group '{group_name}'. "
                f"Available: {list(self.scenario_groups.keys())}"
            )

        return self.scenario_groups[group_name]

    def list_scenarios(self) -> List[str]:
        """List all available scenarios"""
        return list(self.scenarios.keys())

    def list_groups(self) -> List[str]:
        """List all available scenario groups"""
        return list(self.scenario_groups.keys())


# ========== Utility Functions ==========

def load_scenario_config(scenario_name: str) -> Dict[str, Any]:
    """
    Convenience function to load a scenario

    Args:
        scenario_name: Scenario name

    Returns:
        Scenario configuration
    """
    manager = ScenarioManager()
    return manager.load_scenario(scenario_name)


def get_mvp_scenarios() -> List[str]:
    """Get MVP scenario list (for quick testing)"""
    manager = ScenarioManager()
    return manager.get_scenario_group('mvp')


def get_full_scenarios() -> List[str]:
    """Get full benchmark scenario list"""
    manager = ScenarioManager()
    return manager.get_scenario_group('full')


# ========== Testing ==========

if __name__ == '__main__':
    print("=" * 80)
    print("Scenario Manager Test")
    print("=" * 80)

    manager = ScenarioManager()

    print("\nAvailable scenarios:")
    for scenario_name in manager.list_scenarios():
        scenario = manager.scenarios[scenario_name]
        print(f"  - {scenario_name}: {scenario['description']}")

    print("\nAvailable groups:")
    for group_name in manager.list_groups():
        scenarios = manager.get_scenario_group(group_name)
        print(f"  - {group_name}: {scenarios}")

    print("\nTesting nominal scenario:")
    scenario = manager.load_scenario('nominal')
    print(f"  Duration: {scenario['duration_days']} days")
    print(f"  Initial state: {scenario['initial_state']}")

    print("\nTesting shock_load scenario:")
    scenario = manager.load_scenario('shock_load')
    print(f"  Disturbances: {len(scenario['disturbances'])}")
    for dist in scenario['disturbances']:
        print(f"    - {dist['type']} at day {dist['time_days']}")

    print("\nTesting initial state loading:")
    try:
        state = manager.get_initial_state('nominal')
        print(f"  Loaded {len(state)} state variables")
        print(f"  pH: {state.get('pH', 'N/A')}")
        print(f"  S_ac: {state.get('S_ac', 'N/A')}")
    except Exception as e:
        print(f"  Could not load initial state: {e}")

    print("\nScenario Manager ready.")
    print("=" * 80)
