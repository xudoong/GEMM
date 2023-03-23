void scale(double *v, int len, double scale)
{
    for (int i = 0; i < len; i++)
        v[i] *= scale;
}