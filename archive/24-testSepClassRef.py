# Teste Zugriffe / Prinzip Klassen etc
import random


class Vessel:
    type = 'general Starship'
    ID = -1
    starsystem = 'Sun'
    name = ""
    color = 'blue'
    posxyz = [30, 40, 50]
    isDocked = False

    def __init__(self,name = None):
        if name is not None:
            self.name = name
        else:
            self.name = "unknown " + self.type
        self.ID  = str(int(random.random()*10000)).zfill(4)

    def __str__(self):
        return f'Ship "{self.name}" of type "{self.type}" is located in starsystem {self.starsystem} at position {self.posxyz}'

    @property
    def id(self):
        return self.ID

    def giveAttackOrder(self, who, messageOfDeath):
        who.orderAttack(self, messageOfDeath)


class Mothership(Vessel):
    type = "mothership"
    bay_total = 0
    bay_free = bay_total
    bay_content = []

    def __init__(self, name=None, bays=0):
        self.bay_total = bays
        self.bay_free = bays
        super().__init__(name)

    def loadShip(self, ship:Vessel):
        if self.bay_free > 0:
            if ship.isDocked:
                print(f'Ship {ship.ID} ({ship.name}) is already docked.')
                return
            self.bay_content.append(ship)
            self.bay_free -= 1
            ship.isDocked = True
            print(f'Ship {ship.ID} ({ship.name}) is now docked. {self.bay_free} emtpy bays remaining.')
        else:
            print(f'Ship {ship.ID} ({ship.name}) rejected. All {self.bay_total} bays are occupied.')


    def releaseNextShip(self):
        if self.bay_free < self.bay_total:
            ship = self.bay_content.pop()
            self.bay_free += 1
            ship.isDocked = False
            print(f'Ship {ship.ID} ({ship.name}) released. {self.bay_total -self.bay_free } ships remaining in bays.')
            return ship
        else:
            print(f'No ship in bays. All {self.bay_total} bays are empty.')
            return None

class Fighter(Vessel):
    type = 'Fighter'


    def orderAttack(self, inCommand:Vessel, messageOfDeath):
        print(f'In the name of {inCommand.name}: We ({self.name}) are about to kill you. But first, listen closely: {messageOfDeath}.')




m1 = Mothership("AllBaysOutOfOrder")
m2 = Mothership("MS-01", 7)
print(m1)
print(m2)

f1 = Fighter()
f2 = Fighter('Lizard-12')
f3 = Fighter('Herkules')
f4 = Fighter('Hermes')
f5 = Fighter('Zeus')
f6 = Fighter('Adriana')
print(f1)
print(f2)

m2.giveAttackOrder(f2,"It's nothing personally")

m1.loadShip(f1)
m1.loadShip(f1)

m2.loadShip(f1)
m2.loadShip(f2)
m2.loadShip(f2) # doppelte landung
m2.loadShip(f3)
m2.loadShip(f4)
m2.loadShip(f5)
m2.loadShip(f6)

undocked1 = m2.releaseNextShip()
undocked2 = m2.releaseNextShip()
undocked3 = m2.releaseNextShip()
undocked4 = m2.releaseNextShip()
undocked5 = m2.releaseNextShip()
undocked6 = m2.releaseNextShip()


