# ---------------------------------------------------------
# sftext - Scrollable Formatted Text for pygame
# Copyright (c) 2016 Lucas de Morais Siqueira
# Distributed under the GNU Lesser General Public License version 3.
#
#     Support by using, forking, reporting issues and giving feedback:
#     https://https://github.com/LukeMS/sftext/
#
#     Lucas de Morais Siqueira (aka LukeMS)
#     lucas.morais.siqueira@gmail.com
#
#    This file is part of sftext.
#
#    sftext is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    sftext is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with sftext. If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------

import random

import re


def random_color():
    while True:
        x = []
        for i in range(3):
            x.append(random.randint(0, 255))
        if sum(x) > 123:
            break
    return tuple(x)


def colorize(string):
    patterns = [
        "(\.)(?: )([a-zA-Z])",
        "(\.)(?: )([a-zA-Z])",
        "( {8})([a-zA-Z])"
    ]
    for pattern in patterns:
        while re.search(pattern, string, re.I):
            string = re.sub(
                pattern,
                '\\1{}{}{}\\2'.format(
                    "{style}{",
                    "color {}".format(random_color()),
                    "}" + random.choice([
                        "{bold True}",
                        "{italic True}",
                        "{bold True}{italic True}",
                        "", "", ""]) +
                    random.choice([
                        "{underline True}",
                        "{underline False}",
                        "{underline False}",
                        "{underline False}"
                    ]) +
                    " "), string, 1, re.I)
    return string

text = (
    "{align center}{color (255, 123, 123)}{size 40}Lorem ipsum\n\n" +
    "{style}        " +
    "{style}{color " + "{}".format(random_color()) + "}{underline True}L" +
    "{style}{color " + "{}".format(random_color()) + "}{italic True}o" +
    "{style}{color " + "{}".format(random_color()) + "}{underline True}r" +
    "{style}{color " + "{}".format(random_color()) + "}{italic True}e" +
    "{style}{color " + "{}".format(random_color()) + "}{underline True}m" +
    """{style} {style}{italic True}ipsum{style} dolor {style}{color (0, 0, 255)}sit{style} amet, consectetur {style}{color (0, 0, 255)}adipiscing{style} elit. Fusce quis {style}{color (255, 255, 255)}tempor{style} enim, ac {style}{color (255, 255, 255)}pretium{style} diam. In urna lectus, condimentum eget convallis in, hendrerit ut nibh. Nullam tristique elementum sem. Suspendisse volutpat, lacus id eleifend pellentesque, quam risus scelerisque erat, non egestas massa nisi quis est. Morbi viverra elementum nunc, nec blandit leo pretium id. Duis bibendum posuere augue. Mauris risus ex, venenatis non sem euismod, auctor bibendum justo.

        Morbi orci leo, scelerisque a arcu ac, viverra eleifend risus. Nulla ultrices lorem ac rutrum tristique. Etiam sed posuere enim. Nullam sed sollicitudin odio. Mauris a semper ante. Duis nec mauris ipsum. Pellentesque euismod iaculis felis a venenatis. Maecenas tincidunt, erat non pretium eleifend, massa eros aliquam felis, eu dapibus tortor mauris eget tellus. Mauris dapibus fermentum enim nec lacinia. Nam sed velit lacinia, interdum massa sit amet, congue nibh.

        Cras in nisi facilisis, consequat sem sit amet, aliquet ligula. Fusce cursus ante pharetra, mollis nibh ut, mollis ipsum. Proin hendrerit ipsum sit amet purus molestie, eu semper orci euismod. Nulla in faucibus ex. Mauris egestas iaculis ullamcorper. Maecenas sed nisi vitae ipsum feugiat lobortis. Proin lobortis diam ac ligula aliquam elementum. Nullam est diam, gravida id neque id, faucibus eleifend nibh. Integer maximus euismod tellus, a lobortis tellus congue in. Vivamus maximus nulla a sem ornare, nec sollicitudin quam suscipit. Donec quis porta metus. In hac habitasse platea dictumst.

        Suspendisse lacus nisl, pretium vel leo non, aliquam aliquam elit. Aenean et facilisis sapien. Donec congue vel augue vel porta. Donec purus mauris, congue sit amet justo eu, fringilla scelerisque odio. Cras commodo ultricies est. Fusce in pulvinar elit. Aliquam vestibulum efficitur turpis, et tempus massa efficitur mollis. Nulla vel egestas ex. Phasellus placerat egestas imperdiet. Sed lectus massa, tempor at imperdiet ac, blandit at lorem. Pellentesque lacinia dui vel eleifend imperdiet. Morbi faucibus, odio non pulvinar cursus, enim felis rutrum urna, id tempor nibh felis ut nulla. Vestibulum dui dui, rhoncus eget gravida nec, rutrum quis magna. Suspendisse faucibus tortor vel congue vestibulum.

        Morbi et maximus nibh. Sed metus purus, vulputate ac libero sit amet, finibus mattis ex. Integer interdum massa ut risus consectetur luctus. In auctor mi quis dolor viverra accumsan. Nam sed felis quam. Morbi tellus ipsum, porttitor at lacus facilisis, interdum commodo magna. Maecenas imperdiet et orci at dignissim. Aenean imperdiet tristique vestibulum.

        Morbi fermentum malesuada mauris, et feugiat lacus pharetra et. Sed facilisis diam ac quam sodales pulvinar. Morbi efficitur blandit ex, vitae lacinia justo tincidunt nec. In ullamcorper arcu lorem, at lobortis erat molestie nec. Curabitur sit amet magna eu velit egestas lacinia. Nulla hendrerit ex a odio aliquet, eget commodo neque tempor. Duis ac enim facilisis, euismod felis sit amet, ultricies mi. Curabitur et consequat massa.

        Quisque accumsan ipsum id lorem ultricies posuere. In mollis purus luctus ullamcorper fermentum. Nunc rhoncus leo ac ante egestas, nec ornare enim hendrerit. Vivamus aliquet ac turpis ut placerat. Nunc aliquet luctus augue gravida semper. Mauris eget urna ac augue viverra porttitor eget vitae nibh. Nunc sem odio, mattis quis massa ut, scelerisque tincidunt metus. Vestibulum gravida justo libero, quis posuere mi pretium eget. Morbi faucibus tellus vel mauris sagittis, eget porttitor tortor iaculis. In faucibus, ligula sit amet accumsan posuere, diam magna pharetra velit, eu consequat dolor mauris sit amet diam. Ut dignissim molestie turpis, in ullamcorper orci. Integer id venenatis mi. Nullam at facilisis lectus. Sed in commodo dolor. Pellentesque eu felis eget dolor laoreet laoreet.

        Vivamus pretium risus ut tortor dapibus, in pretium arcu condimentum. Nunc congue ultrices erat, non consequat sapien posuere sit amet. Etiam at lacus a lorem rutrum pharetra in non dolor. Integer aliquam ipsum non fringilla volutpat. Nunc non finibus dolor, at aliquet quam. Duis lacinia lacus sit amet semper tristique. Nam ac lectus aliquam est porta euismod. Mauris efficitur ut lectus a tincidunt. Nam dignissim velit vitae aliquet rutrum. Nam rutrum nulla leo, vitae interdum lorem efficitur sit amet. Cras bibendum nulla ipsum, a placerat eros ullamcorper in. Integer tincidunt ligula a tincidunt sodales. Quisque ut quam volutpat, imperdiet arcu sed, placerat sem. Fusce vitae velit sed ligula hendrerit malesuada eu quis sem. Ut blandit dapibus metus, nec euismod leo commodo sed. Praesent ullamcorper leo a tortor sagittis iaculis.

        Integer venenatis nulla eu mattis posuere. Curabitur fringilla, lorem nec viverra bibendum, ipsum dolor mollis ex, ac tincidunt lacus nibh non elit. Vivamus commodo tellus at lorem tempor venenatis. Nunc tincidunt tristique justo nec feugiat. Fusce interdum sed mi in fermentum. Nullam viverra in ipsum sit amet consectetur. Vivamus molestie scelerisque elit at congue. Suspendisse ut finibus nisl, a posuere ex. Curabitur mollis dui dolor, vitae bibendum odio consectetur ut. Aliquam sed nulla faucibus, accumsan nunc vitae, pellentesque eros.

        Aliquam laoreet turpis gravida elit fringilla, non sagittis nulla efficitur. Proin ultrices nibh sed purus dictum viverra. Mauris aliquet ligula quis elit consequat ullamcorper. Vestibulum id tellus dui. Sed et lacus ex. Vivamus vel mauris eros. Mauris non vulputate leo. Nulla facilisi. Nullam ac euismod massa, in convallis tellus. Maecenas et lacus vitae justo accumsan luctus. Curabitur at volutpat mauris, a ultricies dolor. In rhoncus non mi sit amet rutrum.

        Nunc hendrerit sapien et urna dapibus, et tincidunt ipsum dapibus. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Proin rutrum leo ut iaculis eleifend. Aliquam eget auctor nibh. Aliquam quis augue et libero ullamcorper tincidunt. Suspendisse ac nisl dui. Praesent finibus tincidunt orci ut pretium.

        Nunc feugiat, libero vel pretium viverra, magna urna gravida quam, in accumsan turpis velit nec urna. Phasellus viverra molestie diam, vel laoreet sem facilisis eu. Morbi ac placerat leo. Vestibulum maximus congue nulla sit amet interdum. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Donec non libero quis nisi mollis laoreet ut sodales risus. Interdum et malesuada fames ac ante ipsum primis in faucibus. Sed eu augue sit amet libero tempor sagittis eget at enim. Curabitur tempus enim eget arcu semper sollicitudin. Phasellus fermentum, neque ut tempor eleifend, risus eros gravida purus, vitae tincidunt eros arcu quis felis. Fusce eleifend nulla id aliquet rutrum. Nulla et augue ac lacus auctor iaculis. Morbi iaculis, neque at sodales consequat, neque lectus faucibus justo, tempus dignissim massa ante nec tellus. Cras sagittis ut risus vitae pharetra. Nam in fermentum tellus.

        Donec sit amet sagittis sapien. Nulla facilisi. Integer accumsan dapibus sollicitudin. Nunc placerat enim dolor, eu fringilla quam facilisis non. Phasellus nulla dolor, rhoncus a nulla quis, commodo convallis mi. Curabitur mollis pulvinar sem ultricies convallis. Fusce ac efficitur nisl. Vestibulum ex enim, laoreet sed efficitur nec, sodales nec odio. Duis dapibus dui quam, ut lobortis leo pellentesque vitae. Maecenas sit amet lacus sollicitudin, pretium velit ut, sollicitudin elit. Integer sed nunc augue. Morbi eros nulla, ornare non tortor quis, iaculis facilisis dui. Suspendisse dui tellus, ornare vel tortor at, vestibulum ultrices magna. Proin auctor ut lacus eu imperdiet. Suspendisse potenti.

        In at pulvinar risus, sed semper tellus. Cras aliquet a tellus et malesuada. In pharetra mi sed feugiat eleifend. Donec dignissim urna justo, in iaculis urna vestibulum vel. Sed lobortis elementum quam, in tincidunt odio venenatis sed. Etiam ultrices quis tellus id facilisis. In dignissim volutpat nibh vitae rhoncus. Pellentesque in erat rhoncus, varius augue maximus, vestibulum arcu. Donec felis turpis, blandit quis quam ac, aliquam fringilla justo. Aliquam purus justo, commodo nec malesuada eget, maximus eu elit. Sed ac dapibus ligula. Etiam tempus, massa sed maximus cursus, metus sem semper lorem, eget mollis eros ipsum accumsan ligula. Phasellus dapibus sollicitudin risus ac imperdiet. Morbi lacinia velit ut elit gravida pretium. Morbi elementum viverra nunc. Fusce nec auctor enim, ut laoreet risus.

        Donec mollis arcu enim, in finibus ligula tempor maximus. Nam a efficitur est. Nullam felis dolor, vulputate id tempor dictum, pretium eu risus. Mauris facilisis porttitor velit et fringilla. Fusce volutpat feugiat sem, sit amet porttitor enim pellentesque ac. Praesent bibendum faucibus ipsum pellentesque tempor. Etiam vitae nisi risus. Integer nec ornare nibh, id fringilla elit. Morbi molestie leo vel nibh dapibus sollicitudin. Ut dictum, magna ac finibus commodo, nisl felis ornare libero, et sollicitudin est orci in dolor.

        Fusce ac nibh imperdiet, ultricies sem eu, pellentesque orci. Nullam venenatis risus at quam feugiat convallis. Donec sed elementum metus. Duis lacinia tellus orci, vitae efficitur urna accumsan facilisis. Suspendisse pellentesque, magna sed tempus commodo, dui lectus ornare erat, eu malesuada quam metus luctus purus. Nam suscipit iaculis augue eget consectetur. Fusce ultricies ullamcorper elit ut ultrices. Nullam finibus, lorem sit amet facilisis pharetra, sapien nulla finibus arcu, non maximus eros tellus eget enim. Cras commodo vitae est et laoreet. Etiam et ante vel diam malesuada fermentum a ut felis. Praesent a massa vel dui molestie tristique. Morbi sit amet tincidunt est. Proin risus augue, iaculis nec tortor ut, convallis volutpat libero.

        Vestibulum ullamcorper molestie purus vel aliquam. In maximus lorem semper augue hendrerit rutrum. Sed tincidunt, elit quis imperdiet sagittis, massa elit pellentesque felis, sed auctor diam est at massa. Mauris a turpis feugiat, malesuada turpis ut, faucibus mauris. Nunc ut luctus neque, vel vulputate sapien. Donec diam lectus, auctor ut lectus posuere, finibus pulvinar nunc. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus.

        In consectetur euismod est. Curabitur lorem tortor, dignissim eget tellus dictum, vestibulum iaculis massa. Mauris orci libero, placerat sed odio commodo, fringilla interdum justo. Nulla et leo enim. Curabitur ac posuere massa. Pellentesque vel congue velit. Quisque a efficitur ligula. Nullam finibus sed urna a tincidunt. Fusce maximus mauris velit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Nulla magna neque, efficitur at elementum eget, condimentum viverra mi. Morbi in eros nec lacus tincidunt posuere ac nec lorem. Fusce velit dui, interdum id mauris sed, pharetra posuere leo. Aliquam molestie lacinia lorem at porttitor.

        Duis sed euismod mi. Curabitur eleifend fermentum tortor sed dignissim. Integer ex ligula, dapibus at leo ac, finibus blandit nunc. In scelerisque ante sed feugiat mattis. Ut accumsan blandit lacus, sit amet convallis sem luctus sed. Aenean vehicula leo eget turpis ultrices placerat. Aliquam justo velit, congue et mi quis, commodo aliquam nunc. Aliquam ut commodo nibh, sit amet suscipit urna. Fusce varius gravida leo quis accumsan. Nunc tincidunt justo sapien, at ornare neque pellentesque eu.

        Nunc gravida nibh augue, blandit vehicula mi volutpat pharetra. Donec ac mauris purus. Sed egestas nulla id massa porttitor dapibus. Etiam non tellus congue diam aliquet consectetur. Aliquam ac elit suscipit, bibendum enim nec, auctor purus. In venenatis consequat sem sed accumsan. Suspendisse ornare a elit non consectetur. Nam elit nulla, semper et mollis ornare, mattis id sem. Cras eget est libero. Maecenas in elementum nisl. Nulla rhoncus mauris sed leo commodo, tincidunt lobortis nibh mollis. Suspendisse ut sapien nec dolor vulputate sollicitudin ut id dui. Pellentesque nec tellus ac sapien rhoncus eleifend. Duis egestas auctor est non rutrum. Nullam facilisis nisi ac magna vulputate, vel lacinia velit aliquam.
""")
text = colorize(text)
print('done randomly formatting')

if __name__ == '__main__':
    for line in text.splitlines():
        print(line)
