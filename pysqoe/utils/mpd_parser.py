from parse import parse
import xml.etree.ElementTree as et


class MpdParser():
    @staticmethod
    def get_seg_len(manifest):
        # it should be noted that some mpd does not contain maxSegmentDuration entry
        # should we handle such case?
        mpd_tree = et.parse(manifest)
        mpds = (mpd for mpd in mpd_tree.iter(tag='{urn:mpeg:dash:schema:mpd:2011}MPD') 
                    if 'maxSegmentDuration' in mpd.attrib)
        mpd = next(mpds)
        seg_dur = mpd.get('maxSegmentDuration')
        pattern = "PT{:d}H{:d}M{:f}S"
        hour, minute, second = parse(pattern, seg_dur)
        seg_len = int(3600 * hour + 60 * minute + second)
        return seg_len

    @staticmethod
    def get_duration(manifest):
        mpd_tree = et.parse(manifest)
        mpds = (mpd for mpd in mpd_tree.iter(tag='{urn:mpeg:dash:schema:mpd:2011}MPD') 
                    if 'mediaPresentationDuration' in mpd.attrib)
        mpd = next(mpds)
        presentation_dur = mpd.get('mediaPresentationDuration')
        pattern = "PT{:d}H{:d}M{:f}S"
        hour, minute, second = parse(pattern, presentation_dur)
        duration = int(3600 * hour + 60 * minute + second)
        return duration
    
    @staticmethod
    def get_chunk_list(manifest, sample_type):
        tree = et.parse(manifest)
        
        if sample_type.lower() == 'video': 
            adaptation_sets = (elem for elem in tree.iter(tag='{urn:mpeg:dash:schema:mpd:2011}AdaptationSet') if 'par' in elem.attrib)
        elif sample_type.lower() == 'audio':
            adaptation_sets = (elem for elem in tree.iter(tag='{urn:mpeg:dash:schema:mpd:2011}AdaptationSet') if 'par' not in elem.attrib)
        else:
            raise ValueError("Invalid sample type. Sample type should either be 'video' or 'audio'.")

        chunk_list = []
        for adaptation_set in adaptation_sets:
            representations = (rep for rep in adaptation_set if 'id' in rep.attrib)
            for rep in representations:
                chunk_list_at_rep = []
                seg_urls = (seg_url for seg_url in rep.iter(tag='{urn:mpeg:dash:schema:mpd:2011}SegmentURL') if 'media' in seg_url.attrib)
                for segment_url in seg_urls:
                    chunk_list_at_rep.append(segment_url.get('media'))
                chunk_list.append(chunk_list_at_rep)
        
        return chunk_list

    @staticmethod
    def set_chunk_attr(manifest, chunk_list, key, value):
        et.register_namespace('', "urn:mpeg:dash:schema:mpd:2011")
        tree = et.parse(manifest)
        root = tree.getroot()

        for representation, values_rep in zip(chunk_list, value):
            for segment_name, value_seg in zip(representation, values_rep):
                segment_urls = (seg_url for seg_url in tree.iter(tag='{urn:mpeg:dash:schema:mpd:2011}SegmentURL')
                                            if 'media' in seg_url.attrib and seg_url.get('media') == segment_name)
                segment_url = next(segment_urls)
                segment_url.set(key, str(value_seg))

        newfile = open(manifest, 'wb')
        tree = et.ElementTree(root)
        tree.write(newfile, encoding='utf-8', xml_declaration=True)
        newfile.close()
