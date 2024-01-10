'''
이 코드에서 이미지 디코딩 부분을 없애고 이미지를 그대로 반환하도록 변경하려면 decode_tile 함수와 read_tile 함수를 
수정해야 합니다. 다음은 해당 부분을 수정한 코드입니다:
decode_tile 함수를 삭제한 것이 주요 변경 사항입니다. 이제 read_tile 함수는 이미지를 디코딩하지 않고 그대로 읽어서 
Cairo surface에 그립니다. 이렇게 하면 이미지를 디코딩하는 부분이 없어집니다.
'''


static bool read_tile(openslide_t *osr,
                      cairo_t *cr,
                      struct _openslide_level *level,
                      int64_t tile_col, int64_t tile_row,
                      void *arg,
                      GError **err) {
  struct level *l = (struct level *) level;
  TIFF *tiff = arg;

  // tile size
  int64_t tw = l->tiffl.tile_w;
  int64_t th = l->tiffl.tile_h;

  // cache
  g_autoptr(_openslide_cache_entry) cache_entry = NULL;
  uint32_t *tiledata = _openslide_cache_get(osr->cache,
                                            level, tile_col, tile_row,
                                            &cache_entry);
  if (!tiledata) {
    g_autofree uint32_t *buf = NULL;

    // Instead of decoding the tile, we will directly read the raw tile data
    if (!_openslide_tiff_read_tile_data(&l->tiffl, tiff,
                                        &buf, NULL,  // Do not need buflen
                                        tile_col, tile_row,
                                        err)) {
      return false;
    }

    // clip, if necessary
    if (!_openslide_tiff_clip_tile(&l->tiffl, buf,
                                   tile_col, tile_row,
                                   err)) {
      return false;
    }

    // Store the raw tile data directly in the cache
    tiledata = g_steal_pointer(&buf);
    _openslide_cache_put(osr->cache, level, tile_col, tile_row,
                         tiledata, tw * th * 4,
                         &cache_entry);
  }

  // Draw the raw tile data on the Cairo surface
  g_autoptr(cairo_surface_t) surface =
    cairo_image_surface_create_for_data((unsigned char *) tiledata,
                                        CAIRO_FORMAT_ARGB32,
                                        tw, th, tw * 4);
  cairo_set_source_surface(cr, surface, 0, 0);
  cairo_paint(cr);

  return true;
}
