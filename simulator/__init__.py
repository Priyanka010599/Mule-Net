"""MuleNet synthetic three-bank simulator (data-engineer owned).

Submodules:
    config     env-var configuration
    kyc        placeholder KYC document rendering
    generator  seeded dataset builder -> parquet
    replay     transfers.parquet -> Pub/Sub at a speed multiplier
"""
