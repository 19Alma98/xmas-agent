from pydantic import BaseModel, Field
from .recipe import Recipe, RecipeCategory


class MenuSection(BaseModel):
    category: RecipeCategory
    recipes: list[Recipe] = Field(default_factory=list)
    notes: str | None = Field(None)

    def add_recipe(self, recipe: Recipe) -> None:
        if recipe.category.value == self.category.value:
            self.recipes.append(recipe)

    def get_total_prep_time(self) -> int:
        return sum(
            recipe.prep_time_minutes
            for recipe in self.recipes
            if recipe.prep_time_minutes
        )

    def get_total_cook_time(self) -> int:
        return sum(
            recipe.cook_time_minutes
            for recipe in self.recipes
            if recipe.cook_time_minutes
        )


class Menu(BaseModel):
    title: str = Field("Christmas Menu")
    description: str | None = Field(None)
    appetizers: MenuSection = Field(
        default_factory=lambda: MenuSection(category=RecipeCategory.APPETIZER)  # type:ignore
    )
    main_dishes: MenuSection = Field(
        default_factory=lambda: MenuSection(category=RecipeCategory.MAIN_DISH)  # type:ignore
    )
    second_plates: MenuSection = Field(
        default_factory=lambda: MenuSection(category=RecipeCategory.SECOND_PLATE)  # type:ignore
    )
    desserts: MenuSection = Field(
        default_factory=lambda: MenuSection(category=RecipeCategory.DESSERT)  # type:ignore
    )
    number_of_guests: int | None = Field(None)

    summary_notes: str | None = Field(None)

    def get_all_recipes(self) -> list[Recipe]:
        return (
            self.appetizers.recipes
            + self.main_dishes.recipes
            + self.second_plates.recipes
            + self.desserts.recipes
        )

    def get_total_prep_time(self) -> int:
        return (
            self.appetizers.get_total_prep_time()
            + self.main_dishes.get_total_prep_time()
            + self.second_plates.get_total_prep_time()
            + self.desserts.get_total_prep_time()
        )

    def get_total_cook_time(self) -> int:
        return (
            self.appetizers.get_total_cook_time()
            + self.main_dishes.get_total_cook_time()
            + self.second_plates.get_total_cook_time()
            + self.desserts.get_total_cook_time()
        )

    def get_total_recipe_time(self) -> int:
        return self.get_total_prep_time() + self.get_total_cook_time()

    def to_formatted_string(self) -> str:
        """Convert menu to a formatted string."""

        def format_section(title: str, section) -> list[str]:
            lines = [f"{title}", "-" * 40]
            if section.recipes:
                for i, recipe in enumerate(section.recipes, 1):
                    dietary = (
                        ", ".join(recipe.dietary_tags) if recipe.dietary_tags else ""
                    )
                    dietary_str = f" [{dietary}]" if dietary else ""
                    desc = recipe.description
                    if len(desc) > 80:
                        desc = desc[:80] + "..."
                    lines.append(f"  {i}. {recipe.name}{dietary_str}")
                    lines.append(f"     {desc}")
            else:
                lines.append("  No recipes selected yet")

            if section.notes:
                lines.append(f"  📝 Note: {section.notes}")
            return lines

        lines = [
            f"🎄 {self.title} 🎄",
            f"For {self.number_of_guests} guests",
            "",
        ]

        if self.description:
            lines.extend([self.description, ""])

        sections = [
            ("🥗 APPETIZERS", self.appetizers),
            ("🍝 MAIN DISHES", self.main_dishes),
            ("🥩 SECOND PLATES", self.second_plates),
            ("🍰 DESSERTS", self.desserts),
        ]
        for title, section in sections:
            lines.extend(format_section(title, section))
            lines.append("")  # add spacing between sections

        if self.summary_notes:
            lines.extend(["📋 NOTES", "-" * 40, f"  {self.summary_notes}", ""])

        lines.append(
            f"⏱️  Total preparation time: ~{self.get_total_prep_time()} minutes"
        )
        lines.append(f"⏱️  Total cooking time: ~{self.get_total_cook_time()} minutes")

        return "\n".join(lines)
