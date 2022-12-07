import pydantic
from tiled.validation_registration import ValidationError

from .schemas import BatteryChargeMetadata, ExperimentalXASMetadata


def validate_xas_metadata(metadata, structure_family, structure, spec):
    if structure_family != "dataframe":
        raise ValidationError(f"structure_family {structure_family} != dataframe")
    try:
        metadata = ExperimentalXASMetadata.parse_obj(metadata)
    except pydantic.ValidationError as e:
        raise ValidationError(str(e))


def validate_xas_tfy(metadata, structure_family, structure, spec):
    validate_xas_metadata(metadata, structure_family, structure, spec)

    columns = set(structure.macro.columns)

    if not {"energy", "i0", "tfy"}.issubset(columns):
        raise ValidationError(f"columns {columns} must contain i0 and tfy")


def validate_xas_tey(metadata, structure_family, structure, spec):
    validate_xas_metadata(metadata, structure_family, structure, spec)

    columns = set(structure.macro.columns)

    if not {"energy", "i0", "tey"}.issubset(columns):
        raise ValidationError(f"columns {columns} must contain i0 and tey")


def validate_xas_transmission(metadata, structure_family, structure, spec):
    validate_xas_metadata(metadata, structure_family, structure, spec)

    columns = set(structure.macro.columns)

    if not {"energy", "i0", "itrans"}.issubset(columns):
        raise ValidationError(f"columns {columns} must contain i0 and itrans")


def validate_battery_charge_data(metadata, structure_family, structure, spec):
    try:
        metadata = BatteryChargeMetadata.parse_obj(metadata)
    except pydantic.ValidationError as e:
        raise ValidationError(str(e))
