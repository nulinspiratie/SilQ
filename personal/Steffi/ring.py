class Ring():
    def __init__(self, name, material, shape="round", **kwargs):
        self.name = name
        self.material = material
        self.shape = shape

    def impress(self, value):
        shiny = value
        return shiny

    def makes_you_feel(self):
        return "rich and special"


class SauronsRing(Ring):
    def __init__(self, name, material, inscription, **kwargs):
        super().__init__(name, material, **kwargs)

        self.inscription = inscription

    def makes_you_feel(self):
        super().__init__()
        return "special and allmighty"

    def rule_them_all(self, all_humans):
        slaves = all_humans
        return slaves


name = "Saurons Ring"
material = "unknown"
inscription = "A ring to rule them all"
AndreasGroup=['Andrea', 'Arne', 'Guilherme', ...]

SauronsRing(name, material, inscription)
SauronsRing.ruleThemAll(AndreasGroup)