static void Z_dot_ao_dm(double complex *vm, double complex *ao, double complex *dm,
                      int nao, int nocc, int ngrids, char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
    const char TRANS_N = 'N';
    const double complex D0 = 0;
    const double complex D1 = 1;
    
    if (nao <= BOXSIZE)
    {
        zgemm_(&TRANS_N, &TRANS_N, &nocc, &ngrids, &nao,
               &D1, dm, &nocc, ao, &nao, &D0, vm, &nocc);
        return;
    }
    
    char empty[nbas];
    int nbox = (int)((nao-1)/BOXSIZE) + 1;
    int box_id, bas_id, nd, b0, blen;
    int ao_id = 0;
    for (box_id = 0; box_id < nbox; box_id++)
    {
        empty[box_id] = 1;
    }
    
    box_id = 0;
    b0 = BOXSIZE;
    for (bas_id = 0; bas_id < nbas; bas_id++)
    {
        nd = (bas[ANG_OF] * 2 + 1) * bas[NCTR_OF];
        assert(nd < BOXSIZE);
        ao_id += nd;
        empty[box_id] &= !non0table[bas_id];
        if (ao_id == b0)
        {
            box_id++;
            b0 += BOXSIZE;
        }
        else if (ao_id > b0)
        {
            box_id++;
            b0 += BOXSIZE;
            empty[box_id] = !non0table[bas_id];
        }
        bas += BAS_SLOTS;
    }
    
    memset(vm, 0, sizeof(double complex) * ngrids * nocc);
    
    for (box_id = 0; box_id < nbox; box_id++)
    {
        if (!empty[box_id])
        {
            b0 = box_id * BOXSIZE;
            blen = MIN(nao-b0, BOXSIZE);
            zgemm_(&TRANS_N, &TRANS_N, &nocc, &ngrids, &blen,
                   &D1, dm+b0*nocc, &nocc, ao+b0, &nao,
                   &D1, vm, &nocc);
        }
    }
}


/* vm[ngrids,nocc] = ao[ngrids,i] * dm[i,nocc] */
void Z_ao_dm(double complex *vm, double complex *ao, double complex *dm,
                  int nao, int nocc, int ngrids, int blksize, char *non0table,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
    const int nblk = (ngrids+blksize-1) / blksize;
    int ip, ib;
    
#pragma omp parallel default(none) \
shared(vm, ao, dm, nao, nocc, ngrids, blksize, non0table, \
atm, natm, bas, nbas, env) \
private(ip, ib)
#pragma omp for nowait schedule(static)
    for (ib = 0; ib < nblk; ib++) {
        ip = ib * blksize;
        Z_dot_ao_dm(vm+ip*nocc, ao+ip*nao, dm,
                  nao, nocc, MIN(ngrids-ip, blksize),
                  non0table+ib*nbas,
                  atm, natm, bas, nbas, env);
    }
}
