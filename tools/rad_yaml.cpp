/* rad-yaml -- a radiance command line written as the YAML file `radiance --config` reads
 * (core/fleet/config.h, docs/MULTIMODEL.md §4).
 *
 *     rad-yaml --model /models/q.rad --tp 2 --port 8000 > radiance.yaml
 *     radiance --config radiance.yaml
 *
 * The flags are parsed by the engine's own parser first, so a line the server would have refused is
 * refused here with the same message. The file is then loaded back before it is printed, so what
 * comes out is a file the server accepts.
 */
#include "rad_core.h"
#include "fleet/config.h"

#include <cstdio>
#include <string>
#include <vector>

static const char kUsage[] =
    "usage: rad-yaml FLAGS...   print the YAML file these radiance flags configure\n";

int main(int argc, char** argv) {
    if (argc < 2) {
        fputs(kUsage, stderr);
        return 2;
    }
    rad::Config flags;
    const int s = rad::config_parse(argc, argv, &flags);
    if (s == 1) {
        fputs(kUsage, stderr);
        return 0;
    }
    if (s < 0) return 2;

    const std::string y = rad::fleet::yaml_from_args(argc, argv);
    rad::fleet::FleetConfig fc;
    if (rad::fleet::parse(y, "rad-yaml output", &fc) != RAD_OK) {
        fprintf(stderr, "rad-yaml: the file below does not load; please report this command "
                        "line\n%s", y.c_str());
        return 1;
    }
    fputs(y.c_str(), stdout);
    return 0;
}
