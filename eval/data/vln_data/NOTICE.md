# Dataset sources and terms

These are existing benchmark annotations packaged for SimpleVLN evaluation.
They are not newly collected SimpleVLN data. Upstream credits and terms remain
applicable; a code license does not replace the data terms.

- **VLN-CE / R2R-CE:** [VLN-CE](https://github.com/jacobkrantz/VLN-CE) and its
  [dataset documentation](https://jacobkrantz.github.io/vlnce/data), based on the
  [Room-to-Room dataset](https://github.com/peteanderson80/Matterport3DSimulator).
  The [VLN-CE license notice](https://github.com/jacobkrantz/VLN-CE#license) states
  that Matterport3D-based task datasets are distributed under
  [CC BY-NC-SA 3.0 US](https://creativecommons.org/licenses/by-nc-sa/3.0/us/)
  and the [Matterport3D Terms of Use](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).
- **RxR:** [Room-Across-Room](https://github.com/google-research-datasets/RxR),
  adapted to continuous environments by VLN-CE. The original RxR annotations use
  [CC BY 4.0](https://github.com/google-research-datasets/RxR/blob/main/LICENSE);
  a copy is included as [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt).
  The VLN-CE task-data notice and Matterport3D terms also apply to the continuous
  environment data supplied here.
- **Matterport3D:** [Project and access instructions](https://niessner.github.io/Matterport/).
  Scene assets and information derived from them are subject to the
  [Matterport3D Terms of Use](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf),
  including non-commercial academic use. Section 2.4(b) requires the agreement
  or a link with distributions, and recipient acceptance with a retained record
  for distribution of a substantial portion of the dataset. Scene meshes and
  their companion files are therefore obtained separately from the code repository.

The released episode files preserve all existing evaluation episodes and fields.
Only the RxR detailed reference file was reduced to the matching English guide
episode IDs. The manifest records original-file hashes, released-file hashes,
and the transformation. Please cite the upstream benchmark publications when
reporting results; citation entries are available on the linked project pages.
