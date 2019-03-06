# Teste Zugriffe / Prinzip Klassen Super und DLL

class Mothership:
    type = 'Mothership'
    ID = -1
    starsystem = 'Sun'
    name = 'unknown mothership'
    color = 'blue'
    posxyz = [30, 40, 50]

    def __init__(self,name = None):
        if name is not None: self.name = name

    def __str__(self):
        return f'Ship "{self.name}" of type "{self.type}" is located in starsystem {self.starsystem} at position {self.posxyz}'

    @property
    def id(self):
        return self.ID


class Fighter(Mothership):
    type = 'Fighter'
    name = 'unknown Fighter'









m1 = Mothership()
m2 = Mothership("MS-01")
print(m1)
print(m2)

f1 = Fighter()
f2 = Fighter('Lizard-12')
print(f1)
print(f2)
