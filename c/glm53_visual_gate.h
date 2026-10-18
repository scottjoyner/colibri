/* GLM53 optional text-only serving capability. Pure special-token denial.
 * A text-only loader MUST reject visual placeholders, even when no IMAGE
 * frame is attached: accepting one as a normal embedding is unsafe.
 */
#ifndef GLM53_VISUAL_GATE_H
#define GLM53_VISUAL_GATE_H
static int glm53_visual_ids_present(const int visual_ids[6],
                                    const int *tokens,int n) {
    if(!visual_ids || !tokens || n<0)return 1; /* malformed input: deny */
    for(int i=0;i<n;i++)for(int j=0;j<6;j++)
        if(visual_ids[j]>=0 && tokens[i]==visual_ids[j])return 1;
    return 0;
}
#endif
