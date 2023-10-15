import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

LEGEND_COLOUR = {
    "FILL":         ('#FB96BA', ''),
    "TOPSOIL":      ('#50E73C', ''),
    "GRAVEL":       ('#4C004C', ''),
    "SAND":         ('#FFF500', ''),
    "SILT":         ('#6FA8DC', ''),
    "CLAY":         ('#93520D', ''),
    "PEAT":         ('#C6B07E', ''),
    "ORGANIC":      ('#C6B07E', ''),
    "BLDRCBBL":     ('#BCBCBC', ''),
    "CBBLBLDR":     ('#BCBCBC', ''),
    "GRAVELLY":     ('', 'O'),
    "SANDY":        ('', '-'),
    "SILTY":        ('', '..'),
    "CLAYEY":       ('', '++'),
    "BOULDER":      ('', '|'),
    "MUDSTONE":     ('#6AA84F', '..'),
    "CLAYSTONE":    ('#2D6614', '..'),
    "SILTSTONE":    ('#70859C', '..'),
    "SANDSTONE":    ('#F1C232', '-'),
    "VOLCANIC":     ('#E06666', '|'),
    "BRECCIA":      ('#E06666', '/'),
    "DACITE":       ('#DE87C3', '/'),
    "ANDESITE":     ('#6569B9', '/'),
    "QUARTZ":       ('#edf7fb', '/'),
    "PYRITE":       ('#edf7fb', "**"),
    "CONGLOMERATE": ('#BCBCBC', '..'),
    "SEDIMENTARY":  ('#BCBCBC', '--'),
    "METAMORPHIC":  ('#BCBCBC', '\\'),
    "IGNEOUS":      ('#BCBCBC', '|')
}