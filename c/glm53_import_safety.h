/* GLM53-only host import risk gate. Pure decision function, no environment
 * mutation here. On a small or unknown-RAM host, imported-host *per upload*
 * previously caused heavy PSI and no completed response (Joyner 2026-10-17).
 * A deliberate experimental override is distinct from production admission.
 */
#ifndef GLM53_IMPORT_SAFETY_H
#define GLM53_IMPORT_SAFETY_H
#include <string.h>
enum {
    G53_IMPORT_KEEP=0,      /* unchanged (staging absent, enough RAM, import off) */
    G53_IMPORT_DISABLE=1,   /* force staged memcpy rather than host-page import */
    G53_IMPORT_DENY=2,      /* fail closed on unsafe explicit import */
    G53_IMPORT_OVERRIDE=3   /* explicit named experimental override only */
};
static int glm53_import_decision(double host_total_gb,
                                  const char *vulkan,
                                  const char *staged,
                                  const char *imported,
                                  const char *override) {
    if (!vulkan || strcmp(vulkan,"1") || !staged || strcmp(staged,"1"))
        return G53_IMPORT_KEEP;
    /* total <=0 or NaN is unknown -> apply conservative rule. Note comparisons
     * with NaN are false; guard with positive conjunctive comparison. */
    if (host_total_gb>17.179869184 && host_total_gb<1000000.0)
        return G53_IMPORT_KEEP; /* >16GiB system RAM */
    if (!imported || !*imported)return G53_IMPORT_DISABLE;
    if (!strcmp(imported,"0"))return G53_IMPORT_KEEP;
    if (!strcmp(imported,"1") && override && !strcmp(override,"1"))
        return G53_IMPORT_OVERRIDE;
    return G53_IMPORT_DENY;
}
#endif
