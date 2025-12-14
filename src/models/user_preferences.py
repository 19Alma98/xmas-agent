from enum import Enum
from pydantic import BaseModel, Field


class Allergy(str, Enum):
    """Common food allergies."""

    NUTS = "nuts"
    PEANUTS = "peanuts"
    SESAME = "sesame"
    DAIRY = "dairy"
    EGGS = "eggs"
    FISH = "fish"
    SHELLFISH = "shellfish"
    GLUTEN = "gluten"
    WHEAT = "wheat"
    SOY = "soy"


class UserPreferences(BaseModel):
    """Model representing user preferences for Christmas menu."""

    number_of_guests: int = Field(..., ge=0)
    has_vegetarians: bool = Field(default=False)
    vegetarian_count: int = Field(default=0, ge=0)
    has_vegans: bool = Field(default=False)
    vegan_count: int = Field(default=0, ge=0)
    allergies: list[Allergy] = Field(default_factory=list)
    custom_allergies: list[str] = Field(default_factory=list)
    prefer_traditional: bool = Field(default=True)
    max_difficulty: str = Field(default="medium")
    max_prep_time_minutes: int | None = Field(None)
    max_cook_time_minutes: int | None = Field(None)
    additional_notes: str | None = Field(None)

    def get_dietary_requirements_summary(self) -> str:
        """Get a summary of dietary requirements."""
        requirements = []
        if self.has_vegans:
            requirements.append(
                f"{self.vegan_count if self.vegan_count > 0 else 'some'} vegan(s)"
            )
        if self.has_vegetarians:
            requirements.append(
                f"{self.vegetarian_count if self.vegetarian_count > 0 else 'some'} vegetarian(s)"
            )
        if self.allergies:
            requirements.append(
                f"allergies: {', '.join([a for a in self.allergies])}"
            )
        if self.custom_allergies:
            requirements.append(
                f"other allergies: {', '.join([a for a in self.custom_allergies])}"
            )

        return ", ".join(requirements) if requirements else "No special requirements"

    def get_all_allergens(self) -> list[str]:
        allergens: list[str] = [a.value for a in self.allergies]
        allergens.extend(self.custom_allergies)
        return allergens

    class Config:
        use_enum_values = True
