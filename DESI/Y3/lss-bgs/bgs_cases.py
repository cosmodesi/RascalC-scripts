"""Case definitions for DESI Y3 LSS-BGS RascalC covariance runs."""

from __future__ import annotations

from copy import deepcopy

REGIONS = ("SGC", "NGC")

CURRENT_CAMPAIGNS = {
    "nonkp_bright_faint_priority": {
        "compmd": "nonKP",
        "version": "v2",
        "tracer_zranges": {
            "BGS_BRIGHT+FAINT-21.2": [(0.3, 0.5), (0.1, 0.4), (0.0, 0.5)],
            "BGS_BRIGHT+FAINT-21.35": [(0.0, 0.3), (0.3, 0.5), (0.1, 0.4), (0.0, 0.5)],
            "BGS_BRIGHT+FAINT-20.7": [(0.0, 0.3), (0.3, 0.5)],
        },
    },
    "nonkp_meeting_20260521_bright_faint": {
        "compmd": "nonKP",
        "version": "v2",
        "tracer_zranges": {
            "BGS_BRIGHT+FAINT-21.2": [(0.1, 0.3)],
            "BGS_BRIGHT+FAINT-21.35": [(0.1, 0.3)],
            "BGS_BRIGHT+FAINT-20.7": [(0.1, 0.3)],
        },
    },
    "nonkp_meeting_20260521_bright20p7": {
        "compmd": "nonKP",
        "version": "v2",
        "tracer_zranges": {
            "BGS_BRIGHT-20.7": [(0.1, 0.3), (0.1, 0.4), (0.0, 0.5)],
        },
    },
    "nonkp_z0p3_0p45_bright_bf_21p35": {
        "compmd": "nonKP",
        "version": "v2",
        "tracer_zranges": {
            "BGS_BRIGHT-21.35": [(0.3, 0.45)],
            "BGS_BRIGHT+FAINT-21.35": [(0.3, 0.45)],
        },
    },
    "nonkp_bright_compare": {
        "compmd": "nonKP",
        "version": "v2",
        "tracer_zranges": {
            "BGS_BRIGHT-21.35": [(0.1, 0.3), (0.0, 0.3), (0.1, 0.4), (0.0, 0.5)],
            "BGS_BRIGHT-21.2": [(0.1, 0.3), (0.1, 0.4), (0.0, 0.5)],
        },
    },
    "pip_priority": {
        "compmd": "PIP",
        "version": "v2.1",
        "tracer_zranges": {
            "BGS_ANY-21.2": [(0.3, 0.5), (0.1, 0.4), (0.0, 0.5)],
            "BGS_ANY-21.35": [(0.1, 0.3), (0.0, 0.3), (0.3, 0.5), (0.1, 0.4)],
            "BGS_BRIGHT-21.2": [(0.1, 0.3), (0.0, 0.3), (0.1, 0.4), (0.0, 0.5)],
            "BGS_BRIGHT-21.35": [(0.1, 0.3), (0.0, 0.3), (0.1, 0.4), (0.0, 0.5)],
        },
    },
    "pip_20p7": {
        "compmd": "PIP",
        "version": "v2.1",
        "tracer_zranges": {
            "BGS_BRIGHT-20.7": [(0.0, 0.3), (0.0, 0.5), (0.1, 0.4), (0.1, 0.5), (0.3, 0.5)],
            "BGS_ANY-20.7": [(0.0, 0.3), (0.0, 0.5), (0.1, 0.4), (0.1, 0.5), (0.3, 0.5)],
        },
    },
}

LEGACY_TRACER_ZRANGES = {
    "nonKP": {
        "BGS_BRIGHT-21.35": [(0.1, 0.4), (0.0, 0.3)],
        "BGS_BRIGHT+FAINT-21.35": [(0.1, 0.4), (0.0, 0.3), (0.3, 0.5)],
        "BGS_BRIGHT+FAINT-20.7": [(0.0, 0.3), (0.3, 0.5)],
    },
    "PIP": {
        "BGS_BRIGHT-21.35": [(0.1, 0.4), (0.0, 0.3)],
        "BGS_ANY-21.35": [(0.1, 0.4), (0.0, 0.3), (0.3, 0.5)],
        "BGS_ANY-20.7": [(0.0, 0.3), (0.3, 0.5)],
    },
}

CAMPAIGN_CHOICES = ("legacy",) + tuple(CURRENT_CAMPAIGNS)

N_LOOPS = {
    # "-21.5": {(0.1, 0.4): {"SGC": 1536, "NGC": 768}},
    "-21.35": {
        (0.0, 0.5): {"SGC": 1536, "NGC": 1536},
        (0.1, 0.4): {"SGC": 1536, "NGC": 768},
        (0.0, 0.3): {"SGC": 4608, "NGC": 768},
        (0.1, 0.3): {"SGC": 1536, "NGC": 1152},
        (0.25, 0.4): {"SGC": 2048, "NGC": 768},
        (0.3, 0.45): {"SGC": 7680, "NGC": 1536},
        (0.3, 0.5): {"SGC": 2048, "NGC": 768},
    },
    "-21.2": {
        (0.0, 0.5): {"SGC": 2048, "NGC": 768},
        (0.1, 0.4): {"SGC": 1536, "NGC": 768},
        (0.0, 0.3): {"SGC": 1536, "NGC": 768},
        (0.1, 0.3): {"SGC": 2304, "NGC": 1152},
        # (0.25, 0.4): {"SGC": 2048, "NGC": 768},
        (0.3, 0.5): {"SGC": 2048, "NGC": 768},
    },
    "-20.7": {
        (0.0, 0.5): {"SGC": 1536, "NGC": 768},
        (0.1, 0.4): {"SGC": 1536, "NGC": 768},
        (0.0, 0.3): {"SGC": 6912, "NGC": 2304},
        (0.1, 0.3): {"SGC": 7680, "NGC": 2304},
        (0.3, 0.5): {"SGC": 1536, "NGC": 768},
        (0.1, 0.5): {"SGC": 1536, "NGC": 768},
    },
    # "-20.2": {
    #     (0.1, 0.25): {"SGC": 2048, "NGC": 1024},
    #     (0.1, 0.4): {"SGC": 1024, "NGC": 256},
    # },
}


def normalize_zrange(zrange):
    return tuple(float(value) for value in zrange)


def zrange_label(zrange):
    zmin, zmax = normalize_zrange(zrange)
    return f"{zmin:g}-{zmax:g}"


def get_campaign_config(campaign="legacy", compmd=None, version=None):
    if campaign is None:
        campaign = "legacy"
    if campaign == "legacy":
        compmd = compmd or "nonKP"
        version = version or ("v2.1" if compmd == "PIP" else "v2")
        if compmd not in LEGACY_TRACER_ZRANGES:
            raise ValueError(f"Unknown completeness mode for legacy campaign: {compmd}")
        return {
            "name": "legacy",
            "compmd": compmd,
            "version": version,
            "tracer_zranges": deepcopy(LEGACY_TRACER_ZRANGES[compmd]),
        }
    if campaign not in CURRENT_CAMPAIGNS:
        raise ValueError(f"Unknown campaign {campaign!r}; choose one of {', '.join(CAMPAIGN_CHOICES)}")
    config = deepcopy(CURRENT_CAMPAIGNS[campaign])
    if compmd is not None and compmd != config["compmd"]:
        raise ValueError(f"Campaign {campaign!r} uses compmd={config['compmd']}, not {compmd}")
    if version is not None and version != config["version"]:
        raise ValueError(f"Campaign {campaign!r} uses version={config['version']}, not {version}")
    config["name"] = campaign
    return config


def iter_tracer_zranges(campaign="legacy", compmd=None, version=None):
    config = get_campaign_config(campaign=campaign, compmd=compmd, version=version)
    for tracer, zranges in config["tracer_zranges"].items():
        yield tracer, [normalize_zrange(zrange) for zrange in zranges]


def build_cases(campaign="legacy", compmd=None, version=None, regions=REGIONS):
    config = get_campaign_config(campaign=campaign, compmd=compmd, version=version)
    cases = []
    for tracer, zranges in config["tracer_zranges"].items():
        for zrange in zranges:
            for region in regions:
                cases.append({
                    "campaign": config["name"],
                    "compmd": config["compmd"],
                    "version": config["version"],
                    "tracer": tracer,
                    "zrange": normalize_zrange(zrange),
                    "region": region,
                })
    return cases


def array_length(campaign="legacy", compmd=None, version=None):
    return len(build_cases(campaign=campaign, compmd=compmd, version=version))


def case_from_array_id(array_id, campaign="legacy", compmd=None, version=None):
    cases = build_cases(campaign=campaign, compmd=compmd, version=version)
    if array_id < 0 or array_id >= len(cases):
        raise IndexError(f"Array id {array_id} outside valid range 0-{len(cases) - 1}")
    return cases[array_id]


def get_n_loops(phase, tracer, zrange, region, default=None, allow_missing=False):
    loops = N_LOOPS # use the same for simplicity, but could be different for pre/post `phase` if desired
    zrange = normalize_zrange(zrange)
    magcut = "-" + tracer.split("-")[-1] # try to use the same n_loops for ANY/BRIGHT/BRIGHT+FAINT with the same magnitude cut, but could be different if desired
    try:
        return loops[magcut][zrange][region]
    except KeyError:
        if default is not None:
            return int(default)
        if allow_missing:
            return None
        raise KeyError(
            f"No n_loops set for phase={phase}, tracer={tracer}, "
            f"zrange={zrange_label(zrange)}, region={region}. "
            "Set --default-n-loops after checking convergence requirements."
        )


def print_case_summary(campaign="legacy", compmd=None, version=None, phase=None, default_n_loops=None):
    config = get_campaign_config(campaign=campaign, compmd=compmd, version=version)
    cases = build_cases(campaign=campaign, compmd=compmd, version=version)
    array_max = len(cases) - 1
    print(f"# campaign={config['name']} compmd={config['compmd']} version={config['version']}")
    print(f"# array=0-{array_max} n_jobs={len(cases)}")
    if phase is None:
        print("# id region tracer zrange")
        for index, case in enumerate(cases):
            print(f"{index:3d} {case['region']:3s} {case['tracer']:24s} {zrange_label(case['zrange']):8s}")
        return
    print("# id region tracer zrange n_loops")
    for index, case in enumerate(cases):
        n_loops_value = get_n_loops(
            phase, case["tracer"], case["zrange"], case["region"],
            default=default_n_loops, allow_missing=True,
        )
        n_loops = "missing" if n_loops_value is None else str(n_loops_value)
        print(f"{index:3d} {case['region']:3s} {case['tracer']:24s} {zrange_label(case['zrange']):8s} {n_loops}")
